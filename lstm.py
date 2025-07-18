import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt

# ==== Dataset Loader ====
class GestureDataset(Dataset):
    def __init__(self, root_folder):
        self.data = []
        self.labels = []
        self.filenames = []
        self.classes = sorted(os.listdir(root_folder))

        for class_name in self.classes:
            class_path = os.path.join(root_folder, class_name)
            for file in os.listdir(class_path):
                if file.endswith('.npy'):
                    self.data.append(np.load(os.path.join(class_path, file)))
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

# ==== Grouped Attention ====
class GroupedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(3))  # left, right, pose
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

# ==== LSTM Model ====
class GestureLSTM(nn.Module):
    def __init__(self, input_size=150, hidden_size=128, num_classes=15):
        super().__init__()
        self.attn = GroupedAttention()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, return_attn=False):
        x, weights = self.attn(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        if return_attn:
            return out, weights
        return out

# ==== Training Utils ====
def train(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            pred = torch.argmax(output, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def evaluate_loss(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item()
    return total_loss / len(loader)

# ==== Attention Visualization ====
def visualize_attention(model, dataset, index):
    model.eval()
    x, y = dataset[index]
    x_input = x.unsqueeze(0).to(device)

    with torch.no_grad():
        _, attn_weights = model(x_input, return_attn=True)

    sequence = x.numpy()
    coords = sequence.reshape(sequence.shape[0], 50, 3)
    mean_coords = coords.mean(axis=0)
    x_pos = mean_coords[:, 0]
    y_pos = -mean_coords[:, 1]

    attn_array = attn_weights.cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.title(f"Grouped Attention Visualization | Label: {dataset.classes[y]}")
    plt.scatter(x_pos[0:21], y_pos[0:21], c=[attn_array[0]] * 21, cmap='Reds', label='Left Hand', s=100)
    plt.scatter(x_pos[21:42], y_pos[21:42], c=[attn_array[1]] * 21, cmap='Greens', label='Right Hand', s=100)
    plt.scatter(x_pos[42:], y_pos[42:], c=[attn_array[2]] * 8, cmap='Blues', label='Pose', s=100)
    plt.colorbar(label='Attention Weight')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==== Plot Training Progress ====
def plot_training(train_acc, val_acc):
    plt.figure(figsize=(8, 5))
    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==== Main ====
if __name__ == "__main__":
    root_folder = "npy_keypoints"
    dataset = GestureDataset(root_folder)

    video_groups = defaultdict(list)
    for idx, fname in enumerate(dataset.filenames):
        video_id = "_".join(fname.split("_")[:3])
        video_groups[video_id].append(idx)

    video_ids = list(video_groups.keys())
    train_ids, val_ids = train_test_split(video_ids, test_size=0.2, random_state=42)

    train_indices = [i for vid in train_ids for i in video_groups[vid]]
    val_indices = [i for vid in val_ids for i in video_groups[vid]]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureLSTM(input_size=150, hidden_size=128, num_classes=len(dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_acc = 0.0
    train_accs, val_accs = [], []

    for epoch in range(30):
        train_loss = train(model, train_loader, criterion, optimizer)
        val_loss = evaluate_loss(model, val_loader, criterion)
        train_acc = evaluate(model, train_loader)
        val_acc = evaluate(model, val_loader)

        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model_with_attention.pt")
            print(f"✅ Saved new best model at epoch {epoch+1} with accuracy {val_acc:.4f}")

    # Visualize attention after training
    visualize_attention(model, dataset, index=0)

    # Plot accuracy graph
    plot_training(train_accs, val_accs)
