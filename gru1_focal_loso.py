import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Dataset ====
class GestureDataset(Dataset):
    def __init__(self, root_folder):
        self.data, self.labels, self.filenames, self.groups = [], [], [], []
        self.classes = sorted(os.listdir(root_folder))

        for class_name in self.classes:
            path = os.path.join(root_folder, class_name)
            for file in os.listdir(path):
                if file.endswith('.npy'):
                    self.data.append(np.load(os.path.join(path, file)))
                    self.labels.append(class_name)
                    self.filenames.append(file)

                    # ✅ Extract SUBJECT ID for LOSO
                    tokens = file.replace('.npy', '').split('_')
                    if len(tokens) >= 2:
                        subject_id = tokens[1]  # e.g., gesture_dancerID_org
                    else:
                        subject_id = 'unknown'
                    self.groups.append(subject_id)

        self.encoder = LabelEncoder()
        self.labels = self.encoder.fit_transform(self.labels)
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ==== Grouped Attention ====
class GroupedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(3))
        self.norm = nn.LayerNorm(150)

    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B, T, 50, 3)
        x = x.permute(0, 2, 1, 3).reshape(B, 50, T * 3)

        group_weights = F.softmax(self.weights, dim=0)
        left = x[:, 0:21] * group_weights[0]
        right = x[:, 21:42] * group_weights[1]
        pose = x[:, 42:50] * group_weights[2]

        x = torch.cat([left, right, pose], dim=1)
        x = x.view(B, 50, T, 3).permute(0, 2, 1, 3).reshape(B, T, D)
        return self.norm(x), group_weights

# ==== Model ====
class GestureGRU(nn.Module):
    def __init__(self, input_size=150, hidden_size=128, num_classes=52, dropout=0.5, l2_strength=0.001):
        super().__init__()
        self.attn = GroupedAttention()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.l2_strength = l2_strength

    def forward(self, x, return_attn=False):
        x, weights = self.attn(x)
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return (out, weights) if return_attn else out

# ==== Focal Loss ====
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# ==== Train/Eval ====
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        l2_loss = sum(torch.norm(p) for p in model.parameters() if p.requires_grad)
        loss += model.l2_strength * l2_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = torch.argmax(out, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    return correct / total, y_true, y_pred

# ==== LOSO ====
if __name__ == "__main__":
    torch.manual_seed(42)

    root_folder = "npy_keypoints"
    dataset = GestureDataset(root_folder)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCHS = 20
    BATCH_SIZE = 16

    groups = np.array(dataset.groups)
    indices = np.arange(len(dataset))

    logo = LeaveOneGroupOut()

    fold_accs = []
    pooled_true, pooled_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(logo.split(indices, groups=groups)):
        left_out_subjects = np.unique(groups[val_idx])
        print(f"\n===== LOSO Fold {fold+1} | Leaving out subject(s): {left_out_subjects} =====")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE)

        model = GestureGRU(
            input_size=150,
            hidden_size=128,
            num_classes=len(dataset.classes),
            dropout=0.5,
            l2_strength=0.001
        ).to(device)

        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_accs, val_accs = [], []

        for epoch in range(EPOCHS):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            train_acc, _, _ = evaluate(model, train_loader, device)
            val_acc, y_true, y_pred = evaluate(model, val_loader, device)
            val_f1 = f1_score(y_true, y_pred, average='weighted')

            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        fold_accs.append(val_acc)
        pooled_true.extend(y_true)
        pooled_pred.extend(y_pred)

        print(f"✅ Mean LOSO Acc after Fold {fold+1}: {np.mean(fold_accs):.4f}")

    print("\n=== LOSO Final Results ===")
    for i, acc in enumerate(fold_accs, 1):
        print(f"Fold {i} Val Acc: {acc:.4f}")

    print(f"Mean LOSO Acc: {np.mean(fold_accs):.4f}")
    print(f"Std Dev: {np.std(fold_accs):.4f}")

    cm = confusion_matrix(pooled_true, pooled_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(18, 16))
    sns.heatmap(cm_normalized,
                cmap='Blues',
                xticklabels=dataset.encoder.classes_,
                yticklabels=dataset.encoder.classes_,
                annot=False,
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Proportion'})
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.title("Normalized Pooled Confusion Matrix (LOSO)", fontsize=16)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix_heatmap_loso.png", dpi=300)
    plt.show()

    print("📸 Saved: confusion_matrix_heatmap_loso.png")

    print("\n=== Per-Class F1-Scores ===")
    report = classification_report(pooled_true, pooled_pred, target_names=dataset.encoder.classes_, digits=4)
    print(report)

    with open("classification_report_loso.txt", "w") as f:
        f.write(report)

    print("📄 Saved: classification_report_loso.txt")

    # ✅ FINAL: Retrain on ALL data for deployment
    print("\n=== Final Retrain on ALL Data ===")
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    final_model = GestureGRU(
        input_size=150,
        hidden_size=128,
        num_classes=len(dataset.classes),
        dropout=0.5,
        l2_strength=0.001
    ).to(device)

    final_criterion = FocalLoss(alpha=1, gamma=2)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        loss = train(final_model, full_loader, final_criterion, final_optimizer, device)
        print(f"Final Epoch {epoch+1} | Loss: {loss:.4f}")

    torch.save({
        'model_state_dict': final_model.state_dict(),
        'encoder_classes': dataset.encoder.classes_
    }, "final_focal_gru1_loso_model.pth")

    print("✅ Final model saved: final_focal_gru1_loso_model.pth")
