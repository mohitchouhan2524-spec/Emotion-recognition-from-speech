"""
train.py
--------
Training loop for the RAVDESS EmotionCNN model.

Handles:
    - Loss function (CrossEntropyLoss, optionally class-weighted)
    - Optimizer (Adam) + LR scheduler (ReduceLROnPlateau)
    - Epoch loop with train/validation phases
    - Checkpointing (saves best model by validation accuracy)
    - Early stopping (stops if val loss doesn't improve for N epochs)
    - Logging metrics per epoch to a CSV in results/logs/

Usage:
    python src/train.py
    python src/train.py --epochs 50 --batch_size 32 --lr 0.001
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import (
    RANDOM_SEED,
    BEST_MODEL_PATH,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    NUM_CLASSES,
)
from dataset import get_dataloaders
from model import EmotionCNN, count_parameters


def set_seed(seed: int = RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_weights(train_loader, num_classes: int, device) -> torch.Tensor:
    """
    Computes inverse-frequency class weights from the training set so the
    loss function penalizes mistakes on under-represented classes more.
    RAVDESS has fewer 'neutral' samples than other emotions, so this helps.
    """
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(num_classes),
        y=np.array(all_labels),
    )
    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train(
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 5e-4,
    weight_decay: float = 1e-3,
    patience: int = 8,
    use_class_weights: bool = True,
):
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data ---
    train_loader, val_loader, test_loader, label_encoder = get_dataloaders(batch_size=batch_size)

    # --- Model ---
    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)
    print(f"Model parameters: {count_parameters(model):,}")

    # --- Loss ---
    if use_class_weights:
        class_weights = get_class_weights(train_loader, NUM_CLASSES, device)
        print(f"Using class weights: {class_weights.cpu().numpy().round(2)}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # --- Optimizer + Scheduler ---
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # --- Tracking ---
    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = []

    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    print(f"\nStarting training for up to {epochs} epochs (early stopping patience={patience})...\n")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.6f} | {epoch_time:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
        })

        # --- Checkpointing: save best model by validation accuracy ---
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": int(epoch),
                "val_acc": float(val_acc),
                "val_loss": float(val_loss),
                "label_classes": [str(c) for c in label_encoder.classes_],
            }, BEST_MODEL_PATH)
            print(f"  -> New best model saved (val_acc={val_acc:.4f}) to {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1

        # Also save a rolling last-epoch checkpoint (useful to resume training)
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }, os.path.join(CHECKPOINTS_DIR, "last_checkpoint.pt"))

        # --- Early stopping ---
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs "
                  f"({patience} epochs without val_acc improvement).")
            break

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time/60:.1f} minutes. Best val_acc: {best_val_acc:.4f}")

    # --- Save training history log ---
    history_df = pd.DataFrame(history)
    log_path = os.path.join(LOGS_DIR, "training_history.csv")
    history_df.to_csv(log_path, index=False)
    print(f"Training history saved to: {log_path}")

    # --- Final test set evaluation using best model ---
    # weights_only=False is safe here: this checkpoint was generated by this
    # same script in the previous step, not downloaded from an untrusted source.
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal test set performance (best checkpoint): loss={test_loss:.4f}, acc={test_acc:.4f}")

    return model, history_df, label_encoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train the RAVDESS EmotionCNN model.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--no_class_weights", action="store_true",
                         help="Disable class-weighted loss (default: weighting is ON).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        use_class_weights=not args.no_class_weights,
    )