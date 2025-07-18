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
BETA = 0.001  # KL divergence weight

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

# === VAE Encoder ===
class VAEFrameEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, hidden_dim)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        mu, logvar = self.encode(x_flat)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        z = z.view(B, T, -1)
        recon_x = recon_x.view(B, T, -1)
        return z, recon_x, mu, logvar

# === Model ===
class GestureGRUVAE(nn.Module):
    def __init__(self, input_size=150, hidden_size=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.attn = FineGrainedAttention(input_dim=input_size)
        self.encoder = VAEFrameEncoder(input_dim=138, hidden_dim=hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.attn(x)
        z, recon_x, mu, logvar = self.encoder(x)
        out, _ = self.gru(z)
        logits = self.fc(out[:, -1, :])
        return logits, recon_x, x, mu, logvar

# === VAE Loss ===
def vae_loss_function(logits, targets, recon_x, x, mu, logvar, beta=BETA):
    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    ce_loss = F.cross_entropy(logits, targets)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return ce_loss + recon_loss + beta * kld

# === Train / Eval ===
def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, recon_x, x_in, mu, logvar = model(x)
        loss = vae_loss_function(logits, y, recon_x, x_in, mu, logvar)
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
            logits, _, _, _, _ = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    return correct / total, y_true, y_pred

# === MAIN ===
if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GestureDataset(GESTURE_FOLDER)
    groups = np.array(dataset.groups)
    indices = np.arange(len(dataset))

    pooled_true, pooled_pred = [], []
    fold_accs, fold_f1s = [], []
    all_train_curves, all_val_curves = [], []

    print("\n=== Group K-Fold ===")
    gkf = GroupKFold(n_splits=N_SPLITS)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(indices, groups=groups)):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE)

        model = GestureGRUVAE().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_accs, val_accs = [], []
        for epoch in range(EPOCHS):
            loss = train(model, train_loader, optimizer, device)
            train_acc, _, _ = evaluate(model, train_loader, device)
            val_acc, y_true, y_pred = evaluate(model, val_loader, device)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            print(f"Fold {fold+1} | Epoch {epoch+1} | Loss: {loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        pooled_true.extend(y_true)
        pooled_pred.extend(y_pred)
        fold_f1 = f1_score(y_true, y_pred, average='weighted')
        fold_accs.append(val_acc)
        fold_f1s.append(fold_f1)
        print(f"Fold {fold+1} | Final Val Acc: {val_acc:.4f} | F1: {fold_f1:.4f}")

        # Save each fold’s curves
        plt.figure()
        plt.plot(train_accs, label="Train Acc")
        plt.plot(val_accs, label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Fold {fold+1} Train vs Val Acc")
        plt.legend()
        plt.savefig(f"fold_{fold+1}_acc_curve.png")
        plt.close()

        all_train_curves.append(train_accs)
        all_val_curves.append(val_accs)

    # === Plot mean curve ===
    mean_train = np.mean(all_train_curves, axis=0)
    mean_val = np.mean(all_val_curves, axis=0)
    plt.figure()
    plt.plot(mean_train, label="Mean Train Acc")
    plt.plot(mean_val, label="Mean Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Mean K-Fold Train vs Val Acc")
    plt.legend()
    plt.savefig("mean_kfold_acc_curve.png")
    plt.close()

    print(f"\nMean K-Fold Acc: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
    print(f"Mean K-Fold F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

    # === Confusion Matrix ===
    cm = confusion_matrix(pooled_true, pooled_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.title("Pooled Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("pooled_confusion_matrix.png")
    plt.close()

    print(classification_report(pooled_true, pooled_pred, target_names=LABELS))

    # === Final ===
    print("\n=== Final Train on All Data ===")
    final_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    final_model = GestureGRUVAE().to(device)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=0.001)
    for epoch in range(EPOCHS):
        loss = train(final_model, final_loader, optimizer, device)
        print(f"Final Epoch {epoch+1} | Loss: {loss:.4f}")

    torch.save({
        'model_state_dict': final_model.state_dict(),
        'encoder_classes': dataset.encoder.classes_
    }, "final_webcam_gesture_vae.pth")
    print("✅ Final VAE model saved: final_webcam_gesture_vae.pth")
