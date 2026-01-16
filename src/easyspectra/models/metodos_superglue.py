# metodos_superglue.py

import cv2
import numpy as np
import torch
from pathlib import Path
from .matching import Matching


# Weights paths (kept where Matching expects them)
MODELS_PATH = Path("models/weights")
SUPERPOINT_WEIGHTS = MODELS_PATH / "superpoint_v1.pth"
SUPERGLUE_WEIGHTS = MODELS_PATH / "superglue_outdoor.pth"

# Defaults (good starting values for multi/hyperspectral data)
DEFAULTS = {
    # SuperPoint
    "sp_nms_radius": 4,            # suppress very close corners (↑ => fewer keypoints)
    "sp_kpt_threshold": 0.005,     # detection sensitivity (↓ => more keypoints)
    "sp_max_keypoints": 1024,      # maximum number of keypoints

    # SuperGlue
    "sg_match_threshold": 0.20,    # match confidence (↓ => stricter, ↑ => more matches)
    "sg_sinkhorn_iterations": 20,  # iterations of the SuperGlue Sinkhorn solver

    # Homography
    "ransac_thresh": 3.0,          # reprojection error (px). ↑ tolerates more outliers

    # Warp (resampling)
    "warp_interp": "nearest",      # 'nearest' (preserves radiometry), 'linear', 'cubic'
    "border_mode": "replicate",    # 'replicate', 'constant', 'reflect'
}

# OpenCV mappings
_INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "linear":  cv2.INTER_LINEAR,
    "cubic":   cv2.INTER_CUBIC,
}
_BORDER = {
    "replicate": cv2.BORDER_REPLICATE,
    "constant":  cv2.BORDER_CONSTANT,
    "reflect":   cv2.BORDER_REFLECT,
}

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_matching = None


def _frame2tensor(img_gray: np.ndarray, device: str = "cpu"):
    """Convert a (H, W) float32 0–1 array to a tensor (1, 1, H, W)."""
    if img_gray.dtype != np.float32:
        img_gray = img_gray.astype(np.float32)
    if img_gray.max() > 1.0:
        img_gray = img_gray / 255.0
    t = torch.from_numpy(img_gray)[None, None, ...]
    return t.to(device)


def _get_matching(params: dict):
    """
    Build a Matching instance using the current parameters.

    The matcher is (re)created every time, so that any UI-updated knobs
    take effect immediately.
    """
    global _matching

    sp = {
        "nms_radius": int(params.get("sp_nms_radius", DEFAULTS["sp_nms_radius"])),
        "keypoint_threshold": float(params.get("sp_kpt_threshold", DEFAULTS["sp_kpt_threshold"])),
        "max_keypoints": int(params.get("sp_max_keypoints", DEFAULTS["sp_max_keypoints"])),
    }
    sg = {
        "weights": "outdoor",  # use the "outdoor" weights
        "sinkhorn_iterations": int(
            params.get("sg_sinkhorn_iterations", DEFAULTS["sg_sinkhorn_iterations"])
        ),
        "match_threshold": float(
            params.get("sg_match_threshold", DEFAULTS["sg_match_threshold"])
        ),
    }

    matching = Matching({"superpoint": sp, "superglue": sg}).eval().to(_device)

    sp_sd = torch.load(str(SUPERPOINT_WEIGHTS), map_location=_device)
    sg_sd = torch.load(str(SUPERGLUE_WEIGHTS), map_location=_device)
    matching.superpoint.load_state_dict(sp_sd)
    matching.superglue.load_state_dict(sg_sd)

    return matching


def alinhar_por_superglue(
    img_ref: np.ndarray,
    img_mov: np.ndarray,
    params: dict | None = None,
) -> np.ndarray:
    """
    Align img_mov to img_ref using SuperPoint + SuperGlue and a homography.

    Parameters
    ----------
    img_ref : np.ndarray
        Reference image (H, W) or (H, W, C). If 3 channels, it is converted to grayscale.
    img_mov : np.ndarray
        Moving image to be aligned (H, W) or (H, W, C). If 3 channels, it is converted
        to grayscale for matching; the original array is warped.
    params : dict, optional
        Dictionary with optional keys:
            - sp_nms_radius (int)
            - sp_kpt_threshold (float)
            - sp_max_keypoints (int)
            - sg_match_threshold (float)
            - sg_sinkhorn_iterations (int)
            - ransac_thresh (float)
            - warp_interp ('nearest' | 'linear' | 'cubic')
            - border_mode ('replicate' | 'constant' | 'reflect')

    Returns
    -------
    np.ndarray
        Aligned version of img_mov. In case of failure, returns img_mov unchanged.
    """
    try:
        p = {**DEFAULTS, **(params or {})}

        # Prepare grayscale images for keypoint detection
        if img_ref.ndim == 3:
            ref_gray = (
                cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
                if img_ref.shape[2] == 3
                else img_ref[..., 0]
            )
        else:
            ref_gray = img_ref

        if img_mov.ndim == 3:
            mov_gray = (
                cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
                if img_mov.shape[2] == 3
                else img_mov[..., 0]
            )
        else:
            mov_gray = img_mov

        ref_gray = ref_gray.astype(np.float32)
        mov_gray = mov_gray.astype(np.float32)
        if ref_gray.max() > 1.0:
            ref_gray /= 255.0
        if mov_gray.max() > 1.0:
            mov_gray /= 255.0

        Ht, Wt = ref_gray.shape[:2]

        image0 = _frame2tensor(ref_gray, _device)
        image1 = _frame2tensor(mov_gray, _device)
        matching = _get_matching(p)

        with torch.no_grad():
            pred = matching({"image0": image0, "image1": image1})

        kpts0 = pred["keypoints0"][0].detach().cpu().numpy()
        kpts1 = pred["keypoints1"][0].detach().cpu().numpy()
        matches0 = pred["matches0"][0].detach().cpu().numpy()  # (N0,) index in kpts1 or -1

        valid = matches0 > -1
        if valid.sum() < 4:
            print("[SuperGlue] Not enough points to estimate a homography.")
            return img_mov

        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches0[valid]]

        Hmat, mask = cv2.findHomography(
            mkpts1,
            mkpts0,
            cv2.RANSAC,
            float(p["ransac_thresh"]),
        )
        if Hmat is None:
            print("[SuperGlue] Homography could not be estimated.")
            return img_mov

        interp = _INTERP.get(str(p["warp_interp"]).lower(), cv2.INTER_NEAREST)
        border = _BORDER.get(str(p["border_mode"]).lower(), cv2.BORDER_REPLICATE)

        aligned = cv2.warpPerspective(
            img_mov,
            Hmat,
            (Wt, Ht),
            flags=interp,
            borderMode=border,
        )
        return aligned

    except Exception as e:
        print(f"[SuperGlue] Error: {e}")
        return img_mov



