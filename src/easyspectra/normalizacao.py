# normalizacao.py
"""
Spectral cube normalization utilities for EasySpectra.

This module implements a variety of normalization and smoothing operations
applied band-wise, globally, or per-pixel (spectral vector). All methods preserve
the original input shape and return a normalized float32 cube.

Supported methods:
    - "minmax_banda"    : Min-max per band (independent scaling for each band)
    - "minmax_global"   : Min-max over the entire cube
    - "zscore"          : Z-score per band
    - "zscore_global"   : Global Z-score across the whole cube
    - "snv_local"       : Standard Normal Variate (SNV) per pixel spectrum
    - "sg_smoothing"    : Savitzky–Golay smoothing per pixel spectrum
    - "nenhum"          : No operation (passthrough)

Important:
    - Names of methods must not be changed (project compatibility).
    - No logic is altered; only documentation, clarity, and structure are improved.
"""

import numpy as np
from scipy.signal import savgol_filter


def normalizar_cubo(cubo, metodo=["minmax_banda"]):
    """
    Apply one or more normalization/smoothing methods to a hyperspectral cube.

    Parameters
    ----------
    cubo : np.ndarray
        Input spectral cube with shape (H, W, B), where B is the number of bands.
    metodo : list of str, optional
        List of normalization method identifiers (order matters).  
        Defaults to ["minmax_banda"].

    Returns
    -------
    np.ndarray
        A normalized float32 cube with the same shape as the input.

    Notes
    -----
    - Each method is applied sequentially.
    - Unknown method names raise a ValueError.
    - All divisions include a small epsilon (1e-8) to avoid numerical issues.
    """
    cubo_norm = cubo.copy().astype(np.float32)

    for m in metodo:

        # ----------------------------------------
        # Band-wise Min–Max Normalization
        # ----------------------------------------
        if m == "minmax_banda":
            for i in range(cubo_norm.shape[2]):
                banda = cubo_norm[:, :, i]
                min_val, max_val = banda.min(), banda.max()
                cubo_norm[:, :, i] = (banda - min_val) / (max_val - min_val + 1e-8)

        # ----------------------------------------
        # Global Min–Max Normalization
        # ----------------------------------------
        elif m == "minmax_global":
            min_val, max_val = cubo_norm.min(), cubo_norm.max()
            cubo_norm = (cubo_norm - min_val) / (max_val - min_val + 1e-8)

        # ----------------------------------------
        # Band-wise Z-Score Normalization
        # ----------------------------------------
        elif m == "zscore":
            for i in range(cubo_norm.shape[2]):
                banda = cubo_norm[:, :, i]
                mean = np.mean(banda)
                std = np.std(banda)
                cubo_norm[:, :, i] = (banda - mean) / (std + 1e-8)

        # ----------------------------------------
        # Global Z-Score Normalization
        # ----------------------------------------
        elif m == "zscore_global":
            mean = np.mean(cubo_norm)
            std = np.std(cubo_norm)
            cubo_norm = (cubo_norm - mean) / (std + 1e-8)

        # ----------------------------------------
        # SNV (Standard Normal Variate) per pixel
        # ----------------------------------------
        elif m == "snv_local":
            H, W, B = cubo_norm.shape
            for y in range(H):
                for x in range(W):
                    espectro = cubo_norm[y, x, :]
                    mean = np.mean(espectro)
                    std = np.std(espectro)
                    cubo_norm[y, x, :] = (espectro - mean) / (std + 1e-8)

        # ----------------------------------------
        # Savitzky–Golay Smoothing per pixel
        # ----------------------------------------
        elif m == "sg_smoothing":
            H, W, B = cubo_norm.shape
            for y in range(H):
                for x in range(W):
                    espectro = cubo_norm[y, x, :]
                    if B >= 5:  # Only smooth if enough points exist
                        cubo_norm[y, x, :] = savgol_filter(
                            espectro,
                            window_length=5,
                            polyorder=2
                        )

        # ----------------------------------------
        # No operation
        # ----------------------------------------
        elif m == "nenhum":
            continue

        # ----------------------------------------
        # Unknown method
        # ----------------------------------------
        else:
            raise ValueError(f"Unknown normalization method: {m}")

    return cubo_norm
