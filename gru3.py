import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ==== Dataset ====
class GestureDataset(Dataset):
    def __init__(self, root_folder):
        self.data, self.labels, self.filenames = [], [], []
        self.classes = sorted(os.listdir(root_folder))
        for class_name in self.classes:
            path = os.path.join(root_folder, class_name)
            for file in os.listdir(path):
                if file.endswith('.npy'):
                    self.data.append(np.load(os.path.join(path, file)))
                    self.labels.append(class_name)
                    self.filenames.append(file)

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

# ==== Dynamic Grouped Attention ====
class DynamicGroupedAttention(nn.Module):
    def __init__(self, input_dim=150, d_k=32):
        super().__init__()
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
        self.group_names = list(self.groups.keys())
        self.num_groups = len(self.groups)
        self.q_proj = nn.ModuleList([nn.Linear(3, d_k) for _ in self.group_names])
        self.k_proj = nn.ModuleList([nn.Linear(3, d_k) for _ in self.group_names])
        self.v_proj = nn.ModuleList([nn.Linear(3, d_k) for _ in self.group_names])
        self.d_k = d_k
        self.output_norm = nn.LayerNorm(self.num_groups * d_k)

    def forward(self, x):
        B, T, D = x.shape
        K = D // 3
        x = x.view(B, T, K, 3)
        group_outputs = []
        for i, group_name in enumerate(self.group_names):
            indices = self.groups[group_name]
            part = x[:, :, indices, :].reshape(B * T, -1, 3)
            Q = self.q_proj[i](part)
            K_mat = self.k_proj[i](part)
            V = self.v_proj[i](part)
            attn_scores = torch.matmul(Q, K_mat.transpose(-2, -1)) / (self.d_k ** 0.5)
            attn_weights = F.softmax(attn_scores, dim=-1)
            out = torch.matmul(attn_weights, V).mean(dim=1)
            out = out.view(B, T, -1)
            group_outputs.append(out)
        output = torch.cat(group_outputs, dim=-1)
        output = self.output_norm(output)
        return output

# ==== Frame-wise Encoder ====
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

# ==== Full Model ====
class GestureGRU(nn.Module):
    def __init__(self, input_size=150, enc_size=128, hidden_size=128, num_classes=53, dropout=0.5, l2_strength=0.001):
        super().__init__()
        self.attn = DynamicGroupedAttention(input_dim=input_size)
        self.encoder = FrameWiseEncoder(input_dim=self.attn.num_groups * self.attn.d_k + 3, hidden_dim=enc_size)
        self.gru = nn.GRU(enc_size, hidden_size, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.l2_strength = l2_strength

    def forward(self, x):
        B, T, D = x.shape
        K = D // 3
        x = x.view(B, T, K, 3)

        torso_center = x[:, :, 42, :]
        x = x - torso_center.unsqueeze(2)

        right_wrist = x[:, :, 21, :]
        left_wrist = x[:, :, 0, :]
        wrist_vec = right_wrist - left_wrist
        wrist_vec = wrist_vec.view(B, T, -1)

        grouped = self.attn(x.view(B, T, -1))
        combined = torch.cat([grouped, wrist_vec], dim=-1)

        encoded = self.encoder(combined)
        out, _ = self.gru(encoded)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# ==== Utils ====
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in tqdm(loader, desc="Training", leave=False):
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

def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validating", leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            l2_loss = sum(torch.norm(p) for p in model.parameters() if p.requires_grad)
            loss += model.l2_strength * l2_loss
            total_loss += loss.item()
    return total_loss / len(loader)

def evaluate_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating", leave=False):
            x, y = x.to(device), y.to(device)
            pred = torch.argmax(model(x), dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    return correct / total, y_true, y_pred

# ==== Main ====
if __name__ == "__main__":
    root_folder = "npy_keypoints"
    dataset = GestureDataset(root_folder)

    video_groups = defaultdict(list)
    for idx, fname in enumerate(dataset.filenames):
        video_id = "_".join(fname.split("_")[:3])
        video_groups[video_id].append(idx)

    all_groups = list(video_groups.keys())
    gkf = GroupKFold(n_splits=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fold_accuracies = []
    pooled_true, pooled_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(all_groups, groups=all_groups), 1):
        print(f"\n===== Fold {fold} =====")
        train_ids = [all_groups[i] for i in train_idx]
        val_ids = [all_groups[i] for i in val_idx]

        train_samples = [i for vid in train_ids for i in video_groups[vid]]
        val_samples = [i for vid in val_ids for i in video_groups[vid]]

        train_loader = DataLoader(Subset(dataset, train_samples), batch_size=16, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_samples), batch_size=16)

        model = GestureGRU(
            input_size=150,
            enc_size=128,
            hidden_size=64,
            num_classes=len(dataset.classes),
            dropout=0.6,
            l2_strength=0.001
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

        best_acc, patience, trigger = 0.0, 5, 0
        train_losses, val_losses, train_accs, val_accs = [], [], [], []

        for epoch in range(20):
            train_loss = train(model, train_loader, criterion, optimizer)
            val_loss = evaluate_loss(model, val_loader, criterion)
            train_acc, _, _ = evaluate_acc(model, train_loader)
            val_acc, y_true, y_pred = evaluate_acc(model, val_loader)
            scheduler.step(val_acc)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f} | Val Loss = {val_loss:.4f} | Train Acc = {train_acc:.4f} | Val Acc = {val_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f"best_attn_fold{fold}.pt")
                trigger = 0
            else:
                trigger += 1
                if trigger >= patience:
                    break

        fold_accuracies.append(best_acc)
        pooled_true.extend(y_true)
        pooled_pred.extend(y_pred)

        # === Plot Loss Curve ===
        plt.figure()
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Fold {fold} Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"loss_curve_fold{fold}.png")
        plt.show()

        # === Plot Accuracy Curve ===
        plt.figure()
        plt.plot(train_accs, label="Train Acc")
        plt.plot(val_accs, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Fold {fold} Accuracy Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"acc_curve_fold{fold}.png")
        plt.show()

    cm = confusion_matrix(pooled_true, pooled_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_norm, annot=False, cmap='Blues',
                xticklabels=dataset.encoder.classes_,
                yticklabels=dataset.encoder.classes_,
                fmt='.2f')
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("dynamic_attention_confusion_matrix.png", dpi=300)
    plt.show()

    report = classification_report(pooled_true, pooled_pred, target_names=dataset.encoder.classes_, digits=4)
    print(report)
    with open("dynamic_attention_classification_report.txt", "w") as f:
        f.write(report)

    print(f"Mean CV Acc: {np.mean(fold_accuracies):.4f}")
    print(f"Std CV Acc: {np.std(fold_accuracies):.4f}")

    print("\n=== Final Retrain on ALL Data ===")
    full_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    final_model = GestureGRU(
        input_size=150,
        enc_size=128,
        hidden_size=64,
        num_classes=len(dataset.classes),
        dropout=0.6,
        l2_strength=0.001
    ).to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
    final_losses = []
    for epoch in range(20):
        loss = train(final_model, full_loader, criterion, optimizer)
        final_losses.append(loss)
        print(f"Final Epoch {epoch+1} | Loss: {loss:.4f}")
    plt.figure()
    plt.plot(final_losses, label="Final Full Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Final Full Train Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("final_full_train_loss_curve.png")
    plt.show()
    torch.save({'model_state_dict': final_model.state_dict(),
                'encoder_classes': dataset.encoder.classes_},
               "final_dynamic_attention_model.pth")
    print("✅ Final model saved: final_dynamic_attention_model.pth")
