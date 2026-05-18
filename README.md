# Bharatanatyam Viniyoga Gesture Recognition

Real-time, sequence-based gesture recognition for classical Indian dance using deep sequential learning. Built during a research internship at the **CDSAML Lab, PES University**.

Published at an **international SCI-indexed conference, 2026**.

---

## Results

| Model | Accuracy |
|-------|----------|
| GRU (LOSO) | **81.96%** |

LOSO = Leave-One-Subject-Out cross-validation — strictest form of subject-independent evaluation.

---

## What this does

Bharatanatyam *Viniyoga* gestures are hand and body movements with specific semantic meaning in classical Indian dance. No large-scale ML dataset existed for this problem.

This project:
- Builds a **dataset of 11,000+ gesture videos** with structured labeling
- Extracts **skeletal keypoints** (pose + hand landmarks) using MediaPipe
- Trains **GRU and LSTM models** in PyTorch on keypoint sequences
- Evaluates with **LOSO cross-validation** to ensure generalization across dancers
- Compares multiple architectures: GRU, GRU with focal loss, LSTM, VAE, Siamese networks

---

## Tech Stack

- **PyTorch** — model training and evaluation
- **MediaPipe** — pose and hand landmark extraction
- **OpenCV** — video processing
- **NumPy / scikit-learn** — data processing and metrics

---

## Setup

```bash
git clone https://github.com/himaniraoo/summer-internship-cdsaml
cd summer-internship-cdsaml
pip install torch torchvision mediapipe opencv-python numpy scikit-learn matplotlib
```

## Run

```bash
# Extract keypoints
python extract_keypoints.py

# Train with LOSO
python gru1_focal_loso.py

# Real-time inference
python video_test.py
```

