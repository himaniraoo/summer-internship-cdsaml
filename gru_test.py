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
HIDDEN_SIZE = 128
MODEL_PATH = "final_gru1_model.pth"
CONFIDENCE_THRESHOLD = 0.80
HOLD_FRAMES = 5
POSE_IDS = [11, 12, 13, 14, 15, 16, 23, 24]

# ==== Model ====
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

class GestureGRU(nn.Module):
    def __init__(self, input_size=150, hidden_size=128, num_classes=52, dropout=0.5):
        super().__init__()
        self.attn = GroupedAttention()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.attn(x)
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)

# ==== Load Model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
encoder_classes = checkpoint['encoder_classes']

model = GestureGRU(input_size=150, hidden_size=128, num_classes=len(encoder_classes)).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# ==== MediaPipe ====
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

# ==== Utils ====
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

# ==== Live Loop ====
sequence = []
display_label = "Detecting..."
hold_counter = 0
last_pred = None
topk_labels = []
topk_conf = []

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(static_image_mode=False) as holistic:
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
                logits = model(x)
                probs = torch.softmax(logits, dim=1).squeeze()
                topk_conf_tensor, topk_indices = torch.topk(probs, k=3)
                topk_conf = topk_conf_tensor.cpu().numpy()
                topk_labels = [encoder_classes[i] for i in topk_indices.cpu().numpy()]
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

        # Draw top-k box
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
