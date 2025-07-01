import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict

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

# ==== LSTM Model ====
class GestureLSTM(nn.Module):
    def __init__(self, input_size=150, hidden_size=128, num_classes=15, num_layers=1):
        super(GestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# ==== Train ====
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

# ==== Evaluate ====
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


# ==== Main ====
if __name__ == "__main__":
    root_folder = "npy_keypoints"
    dataset = GestureDataset(root_folder)

    # ==== Group by Video ID ====
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

    # ==== Train and Save Best Model ====
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_acc = 0.0

for epoch in range(20):
    train_loss = train(model, train_loader, criterion, optimizer)
    val_loss = evaluate_loss(model, val_loader, criterion)
    train_acc = evaluate(model, train_loader)
    val_acc = evaluate(model, val_loader)

    # ✅ Always log all metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}: Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Train Acc = {train_acc:.4f}, Val Acc = {val_acc:.4f}")

    # ✅ Only save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "new_best_model.pt")
        print(f"✅ Saved new best model at epoch {epoch+1} with accuracy {val_acc:.4f}")
