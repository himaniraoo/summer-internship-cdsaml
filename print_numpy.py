""" import numpy as np

def extract_left_elbow_coords(npy_path):
    data = np.load(npy_path, allow_pickle=True)
    left_elbow_start = 126 + 2 * 3  # start index of left elbow (x, y, z)
    left_elbow_end = left_elbow_start + 3
    coords = data[:, left_elbow_start:left_elbow_end]
    return coords  # shape: (frames, 3)

file1 = '/Users/himanipraneshrao/Desktop/sig/internship/classical/hand-gesture/npy_keypoints/drinking poision/drinking_poison_him_aug19_3.npy'
file2 = '/Users/himanipraneshrao/Desktop/sig/internship/classical/hand-gesture/npy_keypoints/writing/writing_him_aug19_11.npy'

elbow1 = extract_left_elbow_coords(file1)
elbow2 = extract_left_elbow_coords(file2)

print("Left Elbow (file 1):")
print(elbow1)

print("\nLeft Elbow (file 2):")
print(elbow2)
 """










import os
import numpy as np
import csv
from scipy.spatial.distance import euclidean
from tqdm import tqdm

# ==== CONFIG ====
THRESHOLD = 0.4  # Distance threshold
FOLDER_1 = '/Users/himanipraneshrao/Desktop/sig/internship/classical/hand-gesture/npy_keypoints/drinking poision'
FOLDER_2 = '/Users/himanipraneshrao/Desktop/sig/internship/classical/hand-gesture/npy_keypoints/writing'
CSV_FILE = 'similar_frames.csv'

# === Helper: Load All Frames with Metadata ===
def load_all_frames_from_folder(folder_path):
    all_frames = []
    all_meta = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.npy') and '_mask' not in filename:
            filepath = os.path.join(folder_path, filename)
            data = np.load(filepath)

            for frame_idx, frame in enumerate(data):
                all_frames.append(frame)
                all_meta.append((filename, frame_idx))

    return np.array(all_frames), all_meta

# === Compare Folders and Save Similar Frames ===
def compare_folders(folder1, folder2, threshold=0.1, csv_path='similar_frames.csv'):
    print(f"Loading frames from {folder1} and {folder2}...")
    frames1, meta1 = load_all_frames_from_folder(folder1)
    frames2, meta2 = load_all_frames_from_folder(folder2)

    print(f"Total frames in '{folder1}': {len(frames1)}")
    print(f"Total frames in '{folder2}': {len(frames2)}\n")

    similar_pairs = []

    print("Comparing frame-by-frame...")
    for i in tqdm(range(len(frames1))):
        for j in range(len(frames2)):
            dist = euclidean(frames1[i], frames2[j])
            if dist < threshold:
                similar_pairs.append((meta1[i][0], meta1[i][1], meta2[j][0], meta2[j][1], dist))

    # === Print ===
    print(f"\n=== Similar Frames (Distance < {threshold}) ===")
    for f1_name, f1_idx, f2_name, f2_idx, dist in similar_pairs:
        print(f"{f1_name}[frame {f1_idx}] ~ {f2_name}[frame {f2_idx}] => Distance: {dist:.4f}")

    print(f"\nTotal similar pairs found: {len(similar_pairs)}")

    # === Save to CSV ===
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['File 1', 'Frame 1', 'File 2', 'Frame 2', 'Distance'])

        for row in similar_pairs:
            writer.writerow(row)

    print(f"\nResults saved to '{csv_path}'.")

# === Entry Point ===
if __name__ == "__main__":
    compare_folders(FOLDER_1, FOLDER_2, threshold=THRESHOLD, csv_path=CSV_FILE)
