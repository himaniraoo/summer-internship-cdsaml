import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
WINDOW_SIZE = 18
INPUT_SIZE = 150
ENCODED_SIZE = 128
HIDDEN_SIZE = 128
GESTURE_FOLDER = "npy_keypoints"
LABELS = sorted(os.listdir(GESTURE_FOLDER))
NUM_CLASSES = len(LABELS)
EPOCHS = 20
BATCH_SIZE = 16
N_SPLITS = 5

# === Dataset ===
class GestureDataset(Dataset):
    def __init__(self, root_folder):
        self.data, self.labels, self.groups = [], [], []
        self.classes = sorted(os.listdir(root_folder))

        for class_name in self.classes:
            class_path = os.path.join(root_folder, class_name)
            for file in os.listdir(class_path):
                if file.endswith('.npy'):
                    self.data.append(np.load(os.path.join(class_path, file)))
                    self.labels.append(class_name)
                    base = file.rsplit('_', 1)[0]
                    self.groups.append(base)

        self.encoder = LabelEncoder()
        self.labels = self.encoder.fit_transform(self.labels)
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.groups = np.array(self.groups)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# === Attention ===
class FineGrainedAttention(nn.Module):
    def __init__(self, input_dim=150):
        super().__init__()
        self.norm = nn.LayerNorm(138)
        self.groups = {
            'left_wrist': [0],
            'left_thumb': [1, 2, 3, 4],
            'left_index': [5, 6, 7, 8],
            'left_middle': [9, 10, 11, 12],
            'left_ring': [13, 14, 15, 16],
            'left_pinky': [17, 18, 19, 20],
            'right_wrist': [21],
            'right_thumb': [22, 23, 24, 25],
            'right_index': [26, 27, 28, 29],
            'right_middle': [30, 31, 32, 33],
            'right_ring': [34, 35, 36, 37],
            'right_pinky': [38, 39, 40, 41],
            'arms': [42, 43, 44, 45]
        }
        self.attn_weights = nn.Parameter(torch.ones(len(self.groups)))
        self.group_names = list(self.groups.keys())

    def forward(self, x):
        B, T, D = x.shape
        K = D // 3
        x = x.view(B, T, K, 3)
        attn = F.softmax(self.attn_weights, dim=0)
        group_outputs = []
        for i, group_name in enumerate(self.group_names):
            indices = self.groups[group_name]
            part = x[:, :, indices, :].reshape(B, T, -1)
            group_outputs.append(part * attn[i])
        x = torch.cat(group_outputs, dim=-1)
        x = self.norm(x)
        return x, attn

# === Encoder ===
class FrameWiseEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B * T, D)
        x = self.encoder(x)
        x = x.view(B, T, -1)
        return x

# === Model ===
class GestureGRU(nn.Module):
    def __init__(self, input_size=150, enc_size=128, hidden_size=128, num_classes=NUM_CLASSES, l2_strength=0.001):
        super().__init__()
        self.attn = FineGrainedAttention(input_dim=input_size)
        self.encoder = FrameWiseEncoder(input_dim=138, hidden_dim=enc_size)
        self.gru = nn.GRU(enc_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.l2_strength = l2_strength

    def forward(self, x):
        x, _ = self.attn(x)
        x = self.encoder(x)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# === Focal Loss ===
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

# === Train / Eval ===
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
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
        pred = torch.argmax(out, dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = correct / total
    return total_loss / len(loader), acc

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
    acc = correct / total
    return acc, y_true, y_pred

# === MAIN ===
# === MAIN ===
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GestureDataset(GESTURE_FOLDER)
    groups = np.array(dataset.groups)
    indices = np.arange(len(dataset))

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_accs, fold_f1s = [], []
    pooled_true, pooled_pred = [], []

    print("\n=== GROUP K-FOLD ===")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups)):
        print(f"\n===== Fold {fold+1} =====")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE)

        model = GestureGRU().to(device)
        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_accs, val_accs = [], []

        for epoch in range(EPOCHS):
            _, train_acc = train(model, train_loader, criterion, optimizer, device)
            val_acc, y_true, y_pred = evaluate(model, val_loader, device)
            print(f"Fold {fold+1} | Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            train_accs.append(train_acc)
            val_accs.append(val_acc)

        pooled_true.extend(y_true)
        pooled_pred.extend(y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"Fold {fold+1} F1: {f1:.4f}")
        fold_accs.append(val_acc)
        fold_f1s.append(f1)

        plt.figure()
        plt.plot(train_accs, label="Train Acc")
        plt.plot(val_accs, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Fold {fold+1} Train vs Val Accuracy")
        plt.legend()
        plt.savefig(f"fold_{fold+1}_acc_curve.png")
        plt.close()

    mean_acc, std_acc = np.mean(fold_accs), np.std(fold_accs)
    mean_f1, std_f1 = np.mean(fold_f1s), np.std(fold_f1s)

    print(f"\nMean K-Fold Acc: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Mean K-Fold F1: {mean_f1:.4f} ± {std_f1:.4f}")

    report = classification_report(pooled_true, pooled_pred, target_names=LABELS)
    print("\nClassification Report (Pooled K-Fold):\n", report)

    with open("classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(pooled_true, pooled_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.title("Pooled Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("pooled_confusion_matrix.png")
    plt.close()

    print("\n✅ Saved: Train/Val Accuracy curves per fold, pooled CM & report ✅")

    # === FINAL FULL TRAIN ===
    print("\n=== FINAL FULL DATA TRAIN ===")
    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    final_model = GestureGRU().to(device)
    final_criterion = FocalLoss(alpha=1, gamma=2)
    final_optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        full_loss, full_acc = train(final_model, full_loader, final_criterion, final_optimizer, device)
        print(f"Full Train | Epoch {epoch+1} | Loss: {full_loss:.4f} | Train Acc: {full_acc:.4f}")

    torch.save({
        'model_state_dict': final_model.state_dict(),
        'encoder_classes': dataset.encoder.classes_
    }, "final_focal_gru2_model.pth")

    print("\n✅ Final model trained on 100% data saved as final_focal_gru2_model.pth ✅")
