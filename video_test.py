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
GESTURE_FOLDER = "npy_keypoints"
LABELS = sorted(os.listdir(GESTURE_FOLDER))
NUM_CLASSES = len(LABELS)
MODEL_PATH = "final_gru1_model.pth"
CONFIDENCE_THRESHOLD = 0.80
HOLD_FRAMES = 5

VIDEO_PATH = "/Users/himanipraneshrao/Desktop/sig/internship/classical/augmentation/augmented_output/use of arrow/use_of_arrow_mru_aug01.mp4"

# ==== Grouped Attention ====
class GroupedAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(3))
        self.norm = nn.LayerNorm(150)

    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B, T, 50, 3)
        x = x.permute(0, 2, 1, 3).reshape(B, 50, T * 3)
        w = F.softmax(self.weights, dim=0)
        left = x[:, 0:21] * w[0]
        right = x[:, 21:42] * w[1]
        pose = x[:, 42:50] * w[2]
        x = torch.cat([left, right, pose], dim=1)
        x = x.view(B, 50, T, 3).permute(0, 2, 1, 3).reshape(B, T, D)
        return self.norm(x), w

# ==== GRU Model ====
class GestureGRU(nn.Module):
    def __init__(self, input_size=150, hidden_size=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.attn = GroupedAttention()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x, _ = self.attn(x)
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

# ==== Load Model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureGRU().to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ==== Interpolation ====
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

# ==== MediaPipe Setup ====
mp_holistic = mp.solutions.holistic
POSE_IDS = [11, 12, 13, 14, 15, 16, 23, 24]

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

# ==== Draw Landmarks ====
def draw_keypoints(frame, results):
    if results.left_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.pose_landmarks:
        for idx in POSE_IDS:
            lm = results.pose_landmarks.landmark[idx]
            h, w, _ = frame.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)

# ==== Process Video ====
cap = cv2.VideoCapture(VIDEO_PATH)
sequence = []
display_label = "Detecting..."
hold_counter = 0
last_pred = None
conf = 0  # default

with mp_holistic.Holistic(static_image_mode=False, model_complexity=1) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        if w > h:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        draw_keypoints(frame, results)

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
                conf, pred_idx = torch.max(probs, dim=0)
                conf = conf.item()
                pred_label = LABELS[pred_idx.item()]

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

        # ==== Display Label (Top-Center, Black, Big) ====
        label_text = f"{display_label} ({conf:.2f})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.8
        thickness = 6
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
        x_text = (frame.shape[1] - text_width) // 2
        y_text = text_height + 30
        cv2.putText(frame, label_text, (x_text, y_text), font, font_scale, (0, 0, 0), thickness)

        cv2.imshow("Gesture Detection", frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# ==== Pause at End ====
while True:
    cv2.imshow("Gesture Detection", frame)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()

# ==== Pause at End (only if frame is valid) ====
if 'frame' in locals():
    while True:
        cv2.imshow("Gesture Detection", frame)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

