import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
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

# === Dataset ===
class GestureDataset(Dataset):
    def __init__(self, root_folder):
        self.data, self.labels, self.groups = [], [], []
        self.subject_ids = []
        self.classes = sorted(os.listdir(root_folder))

        for class_name in self.classes:
            path = os.path.join(root_folder, class_name)
            for file in os.listdir(path):
                if file.endswith('.npy'):
                    self.data.append(np.load(os.path.join(path, file)))
                    self.labels.append(class_name)

                    # ✅ Get subject ID & sequence base ID
                    parts = file.split('_')
                    if 'real' in file:
                        subject_id = parts[1]  # E.g. applyingKajal_him_real
                    else:
                        subject_id = parts[1]  # E.g. applyingKajal_him_aug01_0.npy → him

                    # ✅ Group all windows of same video together
                    base_sequence = '_'.join(parts[:-1]) if parts[-1].isdigit() else file.replace('.npy', '')

                    self.subject_ids.append(subject_id)
                    self.groups.append(f"{subject_id}_{base_sequence}")

        self.encoder = LabelEncoder()
        self.labels = self.encoder.fit_transform(self.labels)
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.subject_ids = np.array(self.subject_ids)
        self.groups = np.array(self.groups)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# === Attention + Encoder ===
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

# === Train/Eval ===
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

# === LOSO ===
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = GestureDataset(GESTURE_FOLDER)
    unique_subjects = sorted(set(dataset.subject_ids))
    print(f"Subjects for LOSO: {unique_subjects}")

    fold_accs = []
    pooled_true, pooled_pred = [], []

    for fold, test_subject in enumerate(unique_subjects):
        print(f"\n===== LOSO Fold {fold+1}: Leaving {test_subject} Out =====")

        test_idx = np.where(dataset.subject_ids == test_subject)[0]
        train_idx = np.where(dataset.subject_ids != test_subject)[0]

        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE)

        model = GestureGRU().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_accs, val_accs = [], []

        for epoch in range(EPOCHS):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            train_acc, _, _ = evaluate(model, train_loader, device)
            val_acc, y_true, y_pred = evaluate(model, test_loader, device)
            val_f1 = f1_score(y_true, y_pred, average='weighted')

            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {val_acc:.4f} | Test F1: {val_f1:.4f}")

        plt.figure()
        plt.plot(train_accs, label="Train Acc")
        plt.plot(val_accs, label="Test Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"LOSO Fold {fold+1}: Train vs Test Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"loso_curve_fold_{fold+1}.png")
        plt.close()
        print(f"📸 Saved: loso_curve_fold_{fold+1}.png")

        fold_accs.append(val_acc)
        pooled_true.extend(y_true)
        pooled_pred.extend(y_pred)

    print("\n=== LOSO Final Results ===")
    for i, acc in enumerate(fold_accs, 1):
        print(f"Fold {i} Acc: {acc:.4f}")
    print(f"Mean: {np.mean(fold_accs):.4f} | Std: {np.std(fold_accs):.4f}")

    cm = confusion_matrix(pooled_true, pooled_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(18, 16))
    sns.heatmap(cm_normalized,
                cmap='Blues',
                xticklabels=dataset.encoder.classes_,
                yticklabels=dataset.encoder.classes_,
                fmt='.2f')
    plt.xticks(rotation=90)
    plt.title("LOSO Confusion Matrix")
    plt.tight_layout()
    plt.savefig("loso_cm.png")
    plt.show()

    print("📸 Saved: loso_cm.png")

    # ✅ Safe report
    labels_used = np.unique(pooled_true)
    report = classification_report(
        pooled_true, pooled_pred,
        labels=labels_used,
        target_names=dataset.encoder.classes_,
        digits=4
    )
    print(report)
    with open("loso_report.txt", "w") as f:
        f.write(report)

    print("\n=== Final Retrain ===")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = GestureGRU().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(EPOCHS):
        loss = train(model, loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1} | Loss: {loss:.4f}")
    torch.save({
        'model': model.state_dict(),
        'classes': dataset.encoder.classes_
    }, "final_model_loso.pth")
    print("✅ Saved: final_model_loso.pth")
