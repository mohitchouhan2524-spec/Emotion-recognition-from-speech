"""
evaluate.py
-----------
Loads the best saved checkpoint (from train.py) and evaluates it on the
held-out test set. Produces:
    - A full classification report (precision/recall/F1 per emotion)
    - A confusion matrix (raw counts + normalized), saved as a plot
    - A CSV of misclassified samples for error analysis

Usage:
    python src/evaluate.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import BEST_MODEL_PATH, NUM_CLASSES, PLOTS_DIR, REPORTS_DIR
from dataset import get_dataloaders
from model import EmotionCNN


@torch.no_grad()
def get_predictions(model, loader, device):
    """Runs the model over a DataLoader and collects predictions + true labels."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for inputs, labels in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, normalize: bool, save_path: str):
    """Plots and saves a confusion matrix heatmap."""
    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = "Confusion Matrix (Normalized by True Label)"
    else:
        cm_display = cm
        fmt = "d"
        title = "Confusion Matrix (Raw Counts)"

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar=True,
    )
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to: {save_path}")


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at {BEST_MODEL_PATH}. Run train.py first."
        )

    # --- Load data (same split logic as training, via fixed RANDOM_SEED) ---
    _, _, test_loader, label_encoder = get_dataloaders()

    # --- Load model + checkpoint ---
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(val_acc={checkpoint['val_acc']:.4f})")

    class_names = checkpoint.get("label_classes", list(label_encoder.classes_))

    # --- Predict on test set ---
    y_true, y_pred, y_probs = get_predictions(model, test_loader, device)

    # --- Classification report ---
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_path = os.path.join(REPORTS_DIR, "classification_report.csv")
    report_df.to_csv(report_path)

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print(f"Saved classification report to: {report_path}")

    # --- Confusion matrices (raw + normalized) ---
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, normalize=False,
                           save_path=os.path.join(PLOTS_DIR, "confusion_matrix_raw.png"))
    plot_confusion_matrix(cm, class_names, normalize=True,
                           save_path=os.path.join(PLOTS_DIR, "confusion_matrix_normalized.png"))

    # --- Identify most-confused emotion pairs (off-diagonal, excluding self) ---
    cm_no_diag = cm.copy().astype(float)
    np.fill_diagonal(cm_no_diag, 0)
    top_confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm_no_diag[i, j] > 0:
                top_confusions.append((class_names[i], class_names[j], int(cm_no_diag[i, j])))
    top_confusions = sorted(top_confusions, key=lambda x: x[2], reverse=True)[:5]

    print("\n--- Top confused emotion pairs (true -> predicted : count) ---")
    for true_label, pred_label, count in top_confusions:
        print(f"  {true_label:>10} -> {pred_label:<10} : {count}")

    # --- Save misclassified samples for error analysis ---
    misclassified_mask = y_true != y_pred
    misclassified_df = pd.DataFrame({
        "true_label": [class_names[i] for i in y_true[misclassified_mask]],
        "predicted_label": [class_names[i] for i in y_pred[misclassified_mask]],
        "confidence": y_probs[misclassified_mask].max(axis=1),
    })
    misclassified_path = os.path.join(REPORTS_DIR, "misclassified_samples.csv")
    misclassified_df.to_csv(misclassified_path, index=False)
    print(f"\nSaved {len(misclassified_df)} misclassified samples to: {misclassified_path}")

    overall_acc = (y_true == y_pred).mean()
    print(f"\nOverall test accuracy: {overall_acc:.4f}")

    return report_df, cm, class_names


if __name__ == "__main__":
    evaluate_model()