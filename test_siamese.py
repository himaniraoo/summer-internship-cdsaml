import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd

# ==== Siamese Model (same as training) ====
class SiameseNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def forward_one(self, x):
        return self.fc(x)

# ==== Load .npy gesture samples and labels ====
def load_gesture_averages(root_dir):
    gesture_vectors = {}
    for gesture in os.listdir(root_dir):
        gesture_path = os.path.join(root_dir, gesture)
        if not os.path.isdir(gesture_path):
            continue
        vectors = []
        for file in os.listdir(gesture_path):
            if file.endswith(".npy"):
                data = np.load(os.path.join(gesture_path, file))
                if data.ndim > 1:
                    data = np.mean(data, axis=0)
                vectors.append(data)
        if vectors:
            gesture_vectors[gesture] = np.mean(vectors, axis=0)  # class average
    return gesture_vectors

# ==== Generate gesture embeddings using Siamese model ====
def get_embeddings(model, gesture_vectors):
    model.eval()
    embeddings = {}
    for gesture, vector in gesture_vectors.items():
        tensor = torch.tensor(vector, dtype=torch.float32)
        with torch.no_grad():
            embedding = model.forward_one(tensor).numpy()
        embeddings[gesture] = embedding
    return embeddings

# ==== Save similarity matrix to Excel ====
def save_similarity_to_excel(embeddings, filename="gesture_similarity.xlsx"):
    gestures = list(embeddings.keys())
    vectors = np.array([embeddings[g] for g in gestures])
    distance_matrix = pairwise_distances(vectors, metric='euclidean')

    df = pd.DataFrame(distance_matrix, index=gestures, columns=gestures)
    df.to_excel(filename)
    print(f"\n✅ Similarity matrix saved to {filename}")

# ==== Run It ====
root_dir = "npy_keypoints"  # 👈 Change this to your folder path
gesture_vectors = load_gesture_averages(root_dir)

input_dim = list(gesture_vectors.values())[0].shape[0]
model = SiameseNet(input_dim)

# Optional: Load weights
# model.load_state_dict(torch.load("siamese_model.pth"))

embeddings = get_embeddings(model, gesture_vectors)
save_similarity_to_excel(embeddings)
