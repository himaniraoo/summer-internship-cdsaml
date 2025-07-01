import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from scipy.interpolate import interp1d

# ==== CONFIG ====
INPUT_SIZE = 150
WINDOW_SIZE = 18  # ✅ You said you're using 18-frame windows
HIDDEN_SIZE = 128
NUM_LAYERS = 1
LABELS = ['applying_kajal', 'beautiful', 'bees', 'brave', 'capable','chairiot wheel','covering with a cloth','day','decision','drinking poison','drums','dyke impending water','elephant','eyebrows','face'] # 🔁 Replace with your own
MODEL_PATH = "new_best_model.pt"

# ==== MediaPipe Setup ====
mp_holistic = mp.solutions.holistic
POSE_IDS = [11, 12, 13, 14, 15, 16, 23, 24]  # shoulders, elbows, wrists, chest

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

# ==== Extract Keypoints ====
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

# ==== Interpolate NaNs ====
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

# ==== Load Model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GestureLSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_classes=len(LABELS), num_layers=NUM_LAYERS)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ==== Inference with Sticky Prediction ====
sequence = []
last_prediction = None
buffer_count = 0
HOLD_FRAMES = 17 # ⏸️ Number of frames to hold prediction after detecting a change

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(static_image_mode=False) as holistic:
    print("🎥 Starting webcam. Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        # Keep sequence length at WINDOW_SIZE
        if len(sequence) > WINDOW_SIZE:
            sequence.pop(0)

        # Only predict when we have enough frames
        if len(sequence) == WINDOW_SIZE:
            seq_np = interpolate_sequence(np.array(sequence))
            x = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(x)
                pred = torch.argmax(output, dim=1).item()
                label = LABELS[pred]

            # Sticky prediction logic
            if last_prediction != label:
                buffer_count += 1
                if buffer_count >= HOLD_FRAMES:
                    last_prediction = label
                    buffer_count = 0
            else:
                buffer_count = 0  # reset buffer if it's the same label

        # Display
        display_label = last_prediction if last_prediction is not None else "Detecting..."
        cv2.putText(frame, f"Gesture: {display_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        cv2.imshow('Gesture Recognition', frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
