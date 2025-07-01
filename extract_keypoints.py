import cv2
import os
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from scipy.interpolate import interp1d

# ==== CONFIG ====
WINDOW_SIZE = 18
STEP_SIZE = 5
KEYPOINT_DIM = 150  # 126 for hands + 24 for pose
SAVE_PATH = 'npy_keypoints'

# === MediaPipe Setup ===
mp_holistic = mp.solutions.holistic
POSE_IDS = [11, 12, 13, 14, 15, 16, 23, 24]  # shoulders, elbows, wrists, chest

# === Extract Keypoints ===
def extract_keypoints(results):
    keypoints = []

    # Hands
    for hand in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand:
            for lm in hand.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])
        else:
            keypoints.extend([np.nan] * 63)  # 21 keypoints × 3

    # Pose (shoulders, elbows, wrists, chest)
    if results.pose_landmarks:
        for i in POSE_IDS:
            lm = results.pose_landmarks.landmark[i]
            keypoints.extend([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([np.nan] * len(POSE_IDS) * 3)

    return keypoints

# === Interpolate Missing Values Within Sequence ===
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
                seq[:, i] = 0  # fallback
    return seq

# === Sliding Window with Final Interpolation ===
def sliding_windows_with_interpolation(sequence, window_size, step):
    clips = []
    T = len(sequence)

    # Regular full clips
    for start in range(0, T - window_size + 1, step):
        clips.append(sequence[start:start + window_size])

    # Handle the final tail part
    remainder_start = ((T - window_size) // step + 1) * step
    if remainder_start < T:
        tail = sequence[remainder_start:]
        current_len = len(tail)

        # Interpolate tail to match window size
        x_old = np.linspace(0, 1, current_len)
        x_new = np.linspace(0, 1, window_size)
        interpolated = []

        for dim in range(tail.shape[1]):
            y = tail[:, dim]
            f = interp1d(x_old, y, kind='linear', fill_value='extrapolate')
            interpolated_dim = f(x_new)
            interpolated.append(interpolated_dim)

        interpolated = np.stack(interpolated, axis=1)
        clips.append(interpolated)

    return clips

# === Process Single Video ===
def process_video(video_path):
    sequence = []
    cap = cv2.VideoCapture(video_path)

    with mp_holistic.Holistic(static_image_mode=False) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)

    cap.release()
    sequence = interpolate_sequence(sequence)
    return np.array(sequence)

# === Process Dataset ===
def process_dataset(root_folder):
    os.makedirs(SAVE_PATH, exist_ok=True)

    for class_name in os.listdir(root_folder):
        class_path = os.path.join(root_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        save_class_path = os.path.join(SAVE_PATH, class_name)
        os.makedirs(save_class_path, exist_ok=True)

        for filename in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
            if not filename.endswith('.mp4'):
                continue

            video_path = os.path.join(class_path, filename)
            sequence = process_video(video_path)
            clips = sliding_windows_with_interpolation(sequence, WINDOW_SIZE, STEP_SIZE)

            for idx, clip in enumerate(clips):
                save_name = f"{filename[:-4]}_{idx}.npy"
                np.save(os.path.join(save_class_path, save_name), clip)

# === Main Entry Point ===
if __name__ == "__main__":
    process_dataset("../augmentation/augmented_videos/new")  # Replace with your actual path
