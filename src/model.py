"""
model.py
--------
CNN architecture for Speech Emotion Recognition on RAVDESS mel-spectrograms.

Input:  (batch, 1, n_mels, time_steps)   -- single-channel "image" of a mel-spectrogram
Output: (batch, num_classes)             -- raw logits over emotion classes

Architecture: 4 convolutional blocks (Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout)
followed by global average pooling and a small fully-connected classifier head.
Using AdaptiveAvgPool2d before the FC layers means the model doesn't care about
the exact time_steps dimension, so changing DURATION/HOP_LENGTH in config.py
won't break the architecture.

Usage:
    python src/model.py    # quick forward-pass sanity check with dummy input
"""

import os
import sys
import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import NUM_CLASSES, N_MELS


class ConvBlock(nn.Module):
    """One Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout block."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class EmotionCNN(nn.Module):
    """
    CNN classifier for mel-spectrogram-based speech emotion recognition.

    Parameters
    ----------
    num_classes : int
        Number of emotion classes to predict (default: from config.py, currently 8).
    in_channels : int
        Number of input channels (1 for a single mel-spectrogram).
    dropout : float
        Dropout probability used in conv blocks and the classifier head.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, in_channels: int = 3, dropout: float = 0.3):
        # in_channels default is 3 (mel-spectrogram + delta + delta-delta),
        # matching feature_extraction.py's extract_features_sequence output.
        # Set in_channels=1 if INCLUDE_DELTA_FEATURES is disabled in config.py.
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 32, dropout=0.2),   # (n_mels, T) -> (n_mels/2, T/2)
            ConvBlock(32, 64, dropout=0.2),             # -> (n_mels/4, T/4)
            ConvBlock(64, 128, dropout=0.3),            # -> (n_mels/8, T/8)
            ConvBlock(128, 256, dropout=0.3),           # -> (n_mels/16, T/16)
        )

        # Global average pooling collapses the spatial dims to 1x1 regardless
        # of input size, so time_steps/n_mels changes in config.py won't
        # require changing this architecture.
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : Tensor of shape (batch, in_channels, n_mels, time_steps)
        returns : Tensor of shape (batch, num_classes)  -- raw logits (use CrossEntropyLoss)
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity check: run a dummy batch through the model and confirm output shape.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_classes=NUM_CLASSES).to(device)

    batch_size = 8
    dummy_time_steps = 130  # roughly matches ~3s audio at default config settings
    dummy_input = torch.randn(batch_size, 3, N_MELS, dummy_time_steps).to(device)

    output = model(dummy_input)

    print(f"Device: {device}")
    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}  (expected: ({batch_size}, {NUM_CLASSES}))")
    print(f"Trainable parameters: {count_parameters(model):,}")