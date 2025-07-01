import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==== Confusion Matrix Plot ====
def plot_confusion_matrix(model, dataloader, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# ==== Main ====
if __name__ == "__main__":
    root_folder = "npy_keypoints"
    model_path = "new_best_model.pt"

    dataset = GestureDataset(root_folder)

    # Create validation set using grouped video IDs (same logic as before)
    video_groups = defaultdict(list)
    for idx, fname in enumerate(dataset.filenames):
        video_id = "_".join(fname.split("_")[:3])
        video_groups[video_id].append(idx)

    from sklearn.model_selection import train_test_split
    video_ids = list(video_groups.keys())
    train_ids, val_ids = train_test_split(video_ids, test_size=0.2, random_state=42)

    val_indices = [i for vid in val_ids for i in video_groups[vid]]
    val_set = torch.utils.data.Subset(dataset, val_indices)
    val_loader = DataLoader(val_set, batch_size=16)

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureLSTM(input_size=150, hidden_size=128, num_classes=len(dataset.classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Plot confusion matrix
    plot_confusion_matrix(model, val_loader, dataset.encoder.classes_)
