# metodos_dl.py

"""
Deep-learning-based homography alignment for EasySpectra.

This module provides a deep homography estimation backend based on the
architecture used in the paper "Unsupervised Deep Homography Estimation".
The main public entry point is `alinhar_por_deep_homography`, which returns
a version of the moving image aligned to the reference image.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import os

# Path to the pre-trained model (keep variable name for compatibility)
CAMINHO_MODELO = "deep_models/DHE/homography_model.pth"


class DeepHomographyNet(nn.Module):
    """
    Convolutional neural network for deep homography estimation.

    The network expects as input a tensor with 2 channels corresponding to
    a pair of images (reference and moving), both resized to 128x128 and
    normalized to [0, 1]. It outputs 8 values representing the offsets of
    the four corners of the base patch.

    Architecture
    -----------
    - Input: (N, 2, 128, 128)
    - Convolutional feature extractor with max pooling
    - Fully-connected head producing 8 regression outputs
    """

    def __init__(self):
        super(DeepHomographyNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8),
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (N, 2, 128, 128).

        Returns
        -------
        torch.Tensor
            Output tensor with shape (N, 8) representing the 4 corner offsets.
        """
        x = self.features(x)
        x = x.view(-1, 128 * 16 * 16)
        x = self.fc(x)
        return x


def alinhar_por_deep_homography(ref, alvo):
    """
    Align the moving image (`alvo`) to the reference (`ref`) using a
    deep homography estimation network.

    This function:
        1. Resizes both images to 128x128.
        2. Normalizes them to [0, 1].
        3. Stacks them into a 2-channel tensor (ref, alvo).
        4. Runs the pre-trained DeepHomographyNet to predict the corner offsets.
        5. Builds a homography from the predicted corners.
        6. Warps the original-resolution `alvo` to the geometry of `ref`.

    Parameters
    ----------
    ref : np.ndarray
        Reference image (target geometry).
    alvo : np.ndarray
        Moving image to be aligned to `ref`.

    Returns
    -------
    np.ndarray
        Aligned version of `alvo`. If any error occurs (e.g. model file is
        missing, invalid homography, etc.), the original `alvo` is returned.
    """
    try:
        # Pre-processing: resize and normalize to [0, 1]
        ref_resized = cv2.resize(ref, (128, 128)).astype(np.float32) / 255.0
        alvo_resized = cv2.resize(alvo, (128, 128)).astype(np.float32) / 255.0

        # Stack into a 2-channel tensor: (2, H, W) → (1, 2, H, W)
        entrada = np.stack([ref_resized, alvo_resized], axis=0)
        entrada = torch.tensor(entrada).unsqueeze(0)

        # Select device (CPU or CUDA)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and weights
        modelo = DeepHomographyNet().to(device)
        checkpoint = torch.load(
            CAMINHO_MODELO,
            map_location=device,
            weights_only=False,  # kept as in the original logic
        )
        modelo.load_state_dict(checkpoint)
        modelo.eval()

        # Run prediction
        with torch.no_grad():
            entrada = entrada.to(device)
            pred = modelo(entrada).cpu().numpy().reshape(-1, 2)

        # Base and warped corner points (in 128x128 space)
        pts_base = np.array(
            [
                [0, 0],
                [127, 0],
                [127, 127],
                [0, 127],
            ],
            dtype=np.float32,
        )

        pts_warped = pts_base + pred.astype(np.float32)

        # Compute homography and warp the original-resolution image
        H, _ = cv2.findHomography(pts_base, pts_warped)
        aligned = cv2.warpPerspective(alvo, H, (ref.shape[1], ref.shape[0]))

        return aligned

    except Exception as e:
        # Keep the log message in English for international users
        print(f"[DeepHomography] Error: {e}")
        return alvo


