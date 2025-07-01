import numpy as np
import matplotlib.pyplot as plt

# Load saved values
train_losses = np.load("train_losses.npy")
val_losses = np.load("val_losses.npy")
train_accuracies = np.load("train_accuracies.npy")
val_accuracies = np.load("val_accuracies.npy")

plt.figure(figsize=(12, 5))
plt.suptitle("Fold 1 Training Curves", fontsize=14)

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss", color="blue", marker='o')
plt.plot(val_losses, label="Val Loss", color="orange", marker='o')
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Train Acc", color="blue", marker='o')
plt.plot(val_accuracies, label="Val Acc", color="orange", marker='o')
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
