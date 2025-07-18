import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

# === CONFIG ===
DATA_DIR = "npy_sequences_smoothed"
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
SAVE_PATH = "best_model_dual.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 53  # Update if needed
SEED = 42
HELD_OUT_SUBJECT = "him"  # Change to your held-out subject

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# === STEP 1: Compute Max Sequence Length ===
def get_max_seq_len(root):
    T_max = 0
    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path): continue
        for fname in os.listdir(cls_path):
            if fname.endswith('.npy'):
                arr = np.load(os.path.join(cls_path, fname))
                T_max = max(T_max, arr.shape[0])
    return T_max

T_max = get_max_seq_len(DATA_DIR)
print(f"📏 Max sequence length (T_max): {T_max}")

# === STEP 2: Dataset Class ===
class GestureDataset(Dataset):
    def __init__(self, samples, T_max):
        self.samples = samples
        self.T_max = T_max

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        data = np.load(path)
        length = data.shape[0]

        if length < self.T_max:
            pad = np.zeros((self.T_max - length, data.shape[1]))
            data = np.vstack([data, pad])

        mask = np.zeros((self.T_max,), dtype=np.float32)
        mask[:length] = 1.0

        hand = data[:, :126]
        pose = data[:, 126:]

        return {
            'hand': torch.tensor(hand, dtype=torch.float32),
            'pose': torch.tensor(pose, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }

# === STEP 3: LOSO Data Split ===
def load_dataset_splits_loso(data_dir, held_out_subject):
    all_train, all_val = [], []
    label_map = {}
    label_index = 0

    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path): continue

        if class_name not in label_map:
            label_map[class_name] = label_index
            label_index += 1

        files = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith(".npy")]

        train_files = []
        val_files = []

        for f in files:
            fname = os.path.basename(f)
            parts = fname.split('_')
            # 🗂️ Adjust index to match your naming convention
            if len(parts) > 1 and parts[1] == held_out_subject:
                val_files.append((f, label_map[class_name]))
            else:
                train_files.append((f, label_map[class_name]))

        all_train.extend(train_files)
        all_val.extend(val_files)

    return all_train, all_val, label_map

train_samples, val_samples, label_map = load_dataset_splits_loso(DATA_DIR, HELD_OUT_SUBJECT)
print(f"✅ LOSO: Held out '{HELD_OUT_SUBJECT}' | Train: {len(train_samples)} | Val: {len(val_samples)}")

train_dataset = GestureDataset(train_samples, T_max)
val_dataset = GestureDataset(val_samples, T_max)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# === STEP 4: Model ===
class DualGRUModel(nn.Module):
    def __init__(self, input_hand=126, input_pose=24, hidden_size=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.hand_gru = nn.GRU(input_hand, hidden_size, batch_first=True, bidirectional=True)
        self.pose_gru = nn.GRU(input_pose, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size * 4, num_classes)

    def forward(self, hand_seq, pose_seq, mask):
        lengths = mask.sum(dim=1).long()

        hand_packed = nn.utils.rnn.pack_padded_sequence(hand_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        pose_packed = nn.utils.rnn.pack_padded_sequence(pose_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, hand_hidden = self.hand_gru(hand_packed)
        _, pose_hidden = self.pose_gru(pose_packed)

        hand_feat = torch.cat((hand_hidden[0], hand_hidden[1]), dim=1)
        pose_feat = torch.cat((pose_hidden[0], pose_hidden[1]), dim=1)

        fused = torch.cat((hand_feat, pose_feat), dim=1)
        return self.fc(self.dropout(fused))

model = DualGRUModel(num_classes=len(label_map)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# === STEP 5: Training Loop ===
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss, train_correct, total = 0.0, 0, 0

    for batch in train_loader:
        hand = batch['hand'].to(DEVICE)
        pose = batch['pose'].to(DEVICE)
        mask = batch['mask'].to(DEVICE)
        labels = batch['label'].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(hand, pose, mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * hand.size(0)
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        total += hand.size(0)

    train_acc = train_correct / total

    model.eval()
    val_loss, val_correct, total_val = 0.0, 0, 0

    with torch.no_grad():
        for batch in val_loader:
            hand = batch['hand'].to(DEVICE)
            pose = batch['pose'].to(DEVICE)
            mask = batch['mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            outputs = model(hand, pose, mask)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * hand.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            total_val += hand.size(0)

    val_acc = val_correct / total_val if total_val > 0 else 0.0

    print(f"📘 Epoch {epoch+1}/{NUM_EPOCHS} | Train Acc: {train_acc:.3f} | Val Acc: {val_acc:.3f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"✅ Best model updated (Val Acc: {val_acc:.3f})")

print(f"\n🎉 LOSO Training complete. Best Val Accuracy: {best_val_acc:.3f}")
