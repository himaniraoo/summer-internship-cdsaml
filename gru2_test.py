import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import interp1d
import mediapipe as mp

# ==== CONFIG ====
WINDOW_SIZE = 18
INPUT_SIZE = 150
ENCODED_SIZE = 128
HIDDEN_SIZE = 128
GESTURE_FOLDER = "npy_keypoints"
LABELS = sorted(os.listdir(GESTURE_FOLDER))
NUM_CLASSES = len(LABELS)
MODEL_PATH = "final_gru2_model.pth"
CONFIDENCE_THRESHOLD = 0.80
HOLD_FRAMES = 5
POSE_IDS = [11, 12, 13, 14, 15, 16, 23, 24]

# ==== Attention + Encoder Model ====
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
    def __init__(self, input_size=150, enc_size=128, hidden_size=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.attn = FineGrainedAttention(input_dim=input_size)
        self.encoder = FrameWiseEncoder(input_dim=138, hidden_dim=enc_size)
        self.gru = nn.GRU(enc_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.attn(x)
        x = self.encoder(x)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# ==== Load Model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureGRU().to(device)

# 🚩 FIX: Load with weights_only=False
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ==== Keypoint Processing ====
def interpolate_sequence(seq):
    seq = np.array(seq)
    for i in range(seq.shape[1]):
        col = seq[:, i]
        if np.any(np.isnan(col)):
            not_nan = ~np.isnan(col)
            if np.sum(not_nan) > 1:
                f = interp1d(np.where(not_nan)[0], col[not_nan], kind='linear', fill_value='extrapolate')
                seq[:, i] = f(np.arange(seq.shape[0]))
            else:
                seq[:, i] = 0
    return seq

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def extract_keypoints(results):
    keypoints = []
    for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand:
            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([np.nan] * 63)
    if results.pose_landmarks:
        for i in POSE_IDS:
            lm = results.pose_landmarks.landmark[i]
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([np.nan] * len(POSE_IDS) * 3)
    return keypoints

# ==== Live Prediction ====
sequence = []
display_label = "Detecting..."
hold_counter = 0
last_pred = None
topk_labels = []
topk_conf = []

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
    print("🎥 Webcam started. Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        for landmarks in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if landmarks:
                mp_drawing.draw_landmarks(frame, landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.pose_landmarks:
            h, w = frame.shape[:2]
            for i in POSE_IDS:
                lm = results.pose_landmarks.landmark[i]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        if len(sequence) > WINDOW_SIZE:
            sequence.pop(0)

        if len(sequence) == WINDOW_SIZE:
            seq_np = interpolate_sequence(np.array(sequence))
            x = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(x)
                probs = torch.softmax(output, dim=1).squeeze()
                topk_conf_tensor, topk_indices = torch.topk(probs, k=3)
                topk_conf = topk_conf_tensor.cpu().numpy()
                topk_labels = [LABELS[i] for i in topk_indices.cpu().numpy()]
                conf = topk_conf[0]
                pred_label = topk_labels[0]

            if conf > CONFIDENCE_THRESHOLD and pred_label != last_pred:
                hold_counter += 1
                if hold_counter >= HOLD_FRAMES:
                    display_label = pred_label
                    last_pred = pred_label
                    hold_counter = 0
            elif pred_label == last_pred:
                hold_counter = 0
            else:
                hold_counter = 0

        cv2.putText(frame, f"Gesture: {display_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 3)

        # Show top 3 predictions
        h, w = frame.shape[:2]
        box_x, box_y = w - 310, 10
        cv2.rectangle(frame, (box_x, box_y), (w - 10, box_y + 100), (50, 50, 50), -1)
        for i, (label, prob) in enumerate(zip(topk_labels, topk_conf)):
            text = f"{label}: {prob * 100:.1f}%"
            color = (0, 255, 0) if i == 0 else (255, 255, 0)
            cv2.putText(frame, text, (box_x + 10, box_y + 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
