# metodos_alinhamento.py
"""
Alignment methods for EasySpectra.

This module provides multiple alignment backends that all share the same
functional pattern:
    - ref:  reference image (target geometry)
    - alvo: moving image to be aligned to ref
    - params: optional dictionary with backend-specific parameters

On success, each method returns a warped version of `alvo` aligned to `ref`.
On failure, the original `alvo` is returned unchanged.

Backends (internal IDs → commercial label):
    - "orb"       → ORB keypoints
    - "sift"      → SIFT keypoints
    - "akaze"     → AKAZE keypoints
    - "ecc"       → ECC (intensity-based)
    - "template"  → Template matching
    - "mi"        → Mutual information (template-based placeholder)
    - "superglue" → SuperGlue (deep matching)

The public router is `alinhar_imagem`, which selects the method based on the
`metodo` string.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from scipy.stats import entropy

# SuperGlue (already implemented in your project)
from .models.metodos_superglue import alinhar_por_superglue as superglue_match


# -----------------------------
# Helpers: interpolation/border/warp
# -----------------------------
def _map_interp(name: str):
    """
    Map a string interpolation name to the corresponding OpenCV flag.
    Defaults to INTER_NEAREST to better preserve radiometry.
    """
    name = (name or "").lower()
    if name == "linear":
        return cv2.INTER_LINEAR
    if name == "cubic":
        return cv2.INTER_CUBIC
    # Default preserves radiometry as much as possible
    return cv2.INTER_NEAREST


def _map_border(name: str):
    """
    Map a string border name to the corresponding OpenCV border mode.
    """
    name = (name or "").lower()
    if name == "constant":
        return cv2.BORDER_CONSTANT
    if name == "reflect":
        return cv2.BORDER_REFLECT
    return cv2.BORDER_REPLICATE


def _safe_homography(H):
    """
    Basic sanity check for a 3x3 homography matrix.

    Returns
    -------
    bool
        True if H is not None and all entries are finite, False otherwise.
    """
    if H is None:
        return False
    if not np.isfinite(H).all():
        return False
    return True


def _warp_with_H(img, H, out_shape, interp="nearest", border="replicate"):
    """
    Apply cv2.warpPerspective with a homography matrix on the original image.

    This helper preserves the original radiometry by applying the warp directly
    to the original `img`, and not to any normalized/8-bit copy used only for
    feature extraction.

    Parameters
    ----------
    img : np.ndarray
        Input image to be warped.
    H : np.ndarray
        3x3 homography matrix.
    out_shape : tuple
        Target output shape (rows, cols) usually taken from the reference image.
    interp : {"nearest", "linear", "cubic"}, optional
        Interpolation mode (mapped internally to OpenCV flags).
    border : {"replicate", "constant", "reflect"}, optional
        Border handling mode (mapped internally to OpenCV flags).
    """
    flags = _map_interp(interp)
    border_mode = _map_border(border)
    return cv2.warpPerspective(
        img,
        H,
        (out_shape[1], out_shape[0]),
        flags=flags,
        borderMode=border_mode,
    )


def _warp_with_affine(img, M, out_shape, interp="nearest", border="replicate"):
    """
    Apply cv2.warpAffine with a 2x3 affine matrix on the original image.

    Parameters
    ----------
    img : np.ndarray
        Input image to be warped.
    M : np.ndarray
        2x3 affine transform matrix.
    out_shape : tuple
        Target output shape (rows, cols) usually taken from the reference image.
    interp : {"nearest", "linear", "cubic"}, optional
        Interpolation mode (mapped internally to OpenCV flags).
    border : {"replicate", "constant", "reflect"}, optional
        Border handling mode (mapped internally to OpenCV flags).
    """
    flags = _map_interp(interp)
    border_mode = _map_border(border)
    return cv2.warpAffine(
        img,
        M,
        (out_shape[1], out_shape[0]),
        flags=flags,
        borderMode=border_mode,
    )


# -----------------------------
# ORB keypoints
# -----------------------------
def alinhar_por_orb(ref, alvo, params=None, **kwargs):
    """
    Align the moving image (`alvo`) to the reference (`ref`) using ORB keypoints
    and a RANSAC-based homography.

    Parameters
    ----------
    ref : np.ndarray
        Reference image (target geometry).
    alvo : np.ndarray
        Image to be aligned to `ref`.
    params : dict, optional
        ORB and warp configuration. Keys:
            - nfeatures (int)
            - scaleFactor (float)
            - nlevels (int)
            - edgeThreshold (int)
            - patchSize (int)
            - good_match_percent (float)
            - ransac_thresh (float)
            - warp_interp (str)
            - border_mode (str)

    Returns
    -------
    np.ndarray
        Warped version of `alvo` aligned to `ref`, or the original `alvo`
        if alignment fails.
    """
    p = params or {}
    # Parameters with defaults
    nfeatures = int(p.get("nfeatures", 5000))
    scaleFactor = float(p.get("scaleFactor", 1.2))
    nlevels = int(p.get("nlevels", 8))
    edgeThreshold = int(p.get("edgeThreshold", 31))
    patchSize = int(p.get("patchSize", 31))
    good_match_percent = float(p.get("good_match_percent", 0.15))
    ransac_thresh = float(p.get("ransac_thresh", 3.0))
    warp_interp = p.get("warp_interp", "nearest")
    border_mode = p.get("border_mode", "replicate")

    try:
        # Use normalized 8-bit copies only for keypoint extraction
        ref_u8 = cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        alvo_u8 = cv2.normalize(alvo, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        orb = cv2.ORB_create(
            nfeatures=nfeatures,
            scaleFactor=scaleFactor,
            nlevels=nlevels,
            edgeThreshold=edgeThreshold,
            patchSize=patchSize,
        )
        kp1, des1 = orb.detectAndCompute(ref_u8, None)
        kp2, des2 = orb.detectAndCompute(alvo_u8, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return alvo

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = matcher.match(des1, des2)
        if len(matches) < 4:
            return alvo

        matches = sorted(matches, key=lambda x: x.distance)
        keep = max(4, int(len(matches) * max(0.01, min(good_match_percent, 1.0))))
        matches = matches[:keep]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if not _safe_homography(H):
            return alvo

        return _warp_with_H(alvo, H, ref.shape, interp=warp_interp, border=border_mode)
    except Exception:
        return alvo


# -----------------------------
# SIFT keypoints
# -----------------------------
def alinhar_por_sift(ref, alvo, params=None, **kwargs):
    """
    Align the moving image (`alvo`) to the reference (`ref`) using SIFT keypoints
    and a ratio-test-filtered homography.

    Parameters
    ----------
    ref : np.ndarray
        Reference image (target geometry).
    alvo : np.ndarray
        Image to be aligned to `ref`.
    params : dict, optional
        SIFT and warp configuration. Keys:
            - contrastThreshold (float)
            - edgeThreshold (int)
            - nOctaveLayers (int)
            - sigma (float)
            - ratio_test (float)
            - ransac_thresh (float)
            - warp_interp (str)
            - border_mode (str)

    Returns
    -------
    np.ndarray
        Warped version of `alvo` aligned to `ref`, or the original `alvo`
        if alignment fails.
    """
    p = params or {}
    contrastThreshold = float(p.get("contrastThreshold", 0.04))
    edgeThreshold = int(p.get("edgeThreshold", 10))
    nOctaveLayers = int(p.get("nOctaveLayers", 3))
    sigma = float(p.get("sigma", 1.6))
    ratio_test = float(p.get("ratio_test", 0.75))
    ransac_thresh = float(p.get("ransac_thresh", 3.0))
    warp_interp = p.get("warp_interp", "nearest")
    border_mode = p.get("border_mode", "replicate")

    try:
        sift = cv2.SIFT_create(
            contrastThreshold=contrastThreshold,
            edgeThreshold=edgeThreshold,
            nOctaveLayers=nOctaveLayers,
            sigma=sigma,
        )
        ref_u8 = cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        alvo_u8 = cv2.normalize(alvo, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        kp1, des1 = sift.detectAndCompute(ref_u8, None)
        kp2, des2 = sift.detectAndCompute(alvo_u8, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return alvo

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        raw = bf.knnMatch(des1, des2, k=2)
        good = []
        for m in raw:
            if len(m) == 2 and m[0].distance < ratio_test * m[1].distance:
                good.append(m[0])

        if len(good) < 4:
            return alvo

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if not _safe_homography(H):
            return alvo

        return _warp_with_H(alvo, H, ref.shape, interp=warp_interp, border=border_mode)
    except Exception:
        return alvo


# -----------------------------
# AKAZE keypoints
# -----------------------------
def alinhar_por_akaze(ref, alvo, params=None, **kwargs):
    """
    Align the moving image (`alvo`) to the reference (`ref`) using AKAZE keypoints
    with the binary MLDB descriptor (more stable in OpenCV).

    Parameters
    ----------
    ref : np.ndarray
        Reference image (target geometry).
    alvo : np.ndarray
        Image to be aligned to `ref`.
    params : dict, optional
        AKAZE and warp configuration. Keys:
            - threshold (float)
            - descriptor_size (int)   (0 = auto)
            - nOctaves (int)
            - nOctaveLayers (int)
            - ransac_thresh (float)
            - good_match_percent (float)
            - warp_interp (str)
            - border_mode (str)

    Returns
    -------
    np.ndarray
        Warped version of `alvo` aligned to `ref`, or the original `alvo`
        if alignment fails.
    """
    p = params or {}
    threshold = float(p.get("threshold", 0.001))
    descriptor_size = int(p.get("descriptor_size", 0))
    nOctaves = int(p.get("nOctaves", 4))
    nOctaveLayers = int(p.get("nOctaveLayers", 4))
    ransac_thresh = float(p.get("ransac_thresh", 3.0))
    good_match_percent = float(p.get("good_match_percent", 0.20))
    warp_interp = p.get("warp_interp", "nearest")
    border_mode = p.get("border_mode", "replicate")

    try:
        # Only MLDB is safe/standard in AKAZE for OpenCV
        akaze = cv2.AKAZE_create(
            descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
            descriptor_size=descriptor_size,
            threshold=threshold,
            nOctaves=nOctaves,
            nOctaveLayers=nOctaveLayers,
        )
        ref_u8 = cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        alvo_u8 = cv2.normalize(alvo, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        kp1, des1 = akaze.detectAndCompute(ref_u8, None)
        kp2, des2 = akaze.detectAndCompute(alvo_u8, None)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return alvo

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.match(des1, des2)
        if len(matches) < 4:
            return alvo
        matches = sorted(matches, key=lambda x: x.distance)
        keep = max(4, int(len(matches) * max(0.01, min(good_match_percent, 1.0))))
        matches = matches[:keep]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if not _safe_homography(H):
            return alvo

        return _warp_with_H(alvo, H, ref.shape, interp=warp_interp, border=border_mode)
    except Exception:
        return alvo


# -----------------------------
# ECC (intensity-based)
# -----------------------------
def alinhar_por_ecc(ref, alvo, params=None, **kwargs):
    """
    Align the moving image (`alvo`) to the reference (`ref`) using OpenCV's
    ECC (Enhanced Correlation Coefficient) algorithm.

    Parameters
    ----------
    ref : np.ndarray
        Reference image (target geometry).
    alvo : np.ndarray
        Image to be aligned to `ref`.
    params : dict, optional
        ECC and warp configuration. Keys:
            - warp_mode ({"translation","euclidean","affine","homography"})
            - number_of_iterations (int)
            - termination_eps (float)
            - gaussFiltSize (int)
            - warp_interp (str)
            - border_mode (str)

    Returns
    -------
    np.ndarray
        Warped version of `alvo` aligned to `ref`, or the original `alvo`
        if alignment fails.
    """
    p = params or {}
    # Map warp mode name to OpenCV constant
    warp_mode_name = (p.get("warp_mode", "affine") or "affine").lower()
    if warp_mode_name == "translation":
        warp_mode = cv2.MOTION_TRANSLATION
    elif warp_mode_name == "euclidean":
        warp_mode = cv2.MOTION_EUCLIDEAN
    elif warp_mode_name == "homography":
        warp_mode = cv2.MOTION_HOMOGRAPHY
    else:
        warp_mode = cv2.MOTION_AFFINE

    number_of_iterations = int(p.get("number_of_iterations", 100))
    termination_eps = float(p.get("termination_eps", 1e-6))
    gaussFiltSize = int(p.get("gaussFiltSize", 0))
    warp_interp = p.get("warp_interp", "nearest")
    border_mode = p.get("border_mode", "replicate")

    try:
        ref_u8 = cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        alvo_u8 = cv2.normalize(alvo, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        if gaussFiltSize and gaussFiltSize > 1:
            # force odd kernel size
            k = int(gaussFiltSize) | 1
            ref_u8 = cv2.GaussianBlur(ref_u8, (k, k), 0)
            alvo_u8 = cv2.GaussianBlur(alvo_u8, (k, k), 0)

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations,
                termination_eps,
            )
            try:
                cc, warp_matrix = cv2.findTransformECC(ref_u8, alvo_u8, warp_matrix, warp_mode, criteria)
            except cv2.error:
                return alvo
            if not _safe_homography(warp_matrix):
                return alvo
            return _warp_with_H(alvo, warp_matrix, ref.shape, interp=warp_interp, border=border_mode)

        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations,
                termination_eps,
            )
            try:
                cc, warp_matrix = cv2.findTransformECC(ref_u8, alvo_u8, warp_matrix, warp_mode, criteria)
            except cv2.error:
                return alvo
            return _warp_with_affine(alvo, warp_matrix, ref.shape, interp=warp_interp, border=border_mode)

    except Exception:
        return alvo


# -----------------------------
# Template matching
# -----------------------------
_TM_MAP = {
    "tm_ccoeff_normed": cv2.TM_CCOEFF_NORMED,
    "tm_ccorr_normed": cv2.TM_CCORR_NORMED,
    "tm_sqdiff": cv2.TM_SQDIFF,
    "tm_sqdiff_normed": cv2.TM_SQDIFF_NORMED,
    "tm_ccoeff": cv2.TM_CCOEFF,
    "tm_ccorr": cv2.TM_CCORR,
}


def alinhar_por_template(ref, alvo, params=None, **kwargs):
    """
    Align the moving image (`alvo`) to the reference (`ref`) using template
    matching and a pure translation model.

    The method finds the best matching location of `ref` inside `alvo` at an
    optional reduced resolution, then applies the resulting translation to
    the original-resolution image.

    Parameters
    ----------
    ref : np.ndarray
        Reference image (template, target geometry).
    alvo : np.ndarray
        Image to be aligned to `ref`.
    params : dict, optional
        Template-matching and warp configuration. Keys:
            - method (str)           (e.g., "TM_CCOEFF_NORMED")
            - resize_factor (float)  (0.2–1.0: lower = faster, coarser)
            - warp_interp (str)
            - border_mode (str)

    Returns
    -------
    np.ndarray
        Warped version of `alvo` aligned to `ref`, or the original `alvo`
        if alignment fails.
    """
    p = params or {}
    method_name = (p.get("method", "TM_CCOEFF_NORMED") or "TM_CCOEFF_NORMED").lower()
    resize_factor = float(p.get("resize_factor", 1.0))
    warp_interp = p.get("warp_interp", "nearest")  # kept for consistency if UI provides it
    border_mode = p.get("border_mode", "replicate")

    try:
        # Optional downscaling to speed up matching
        if 0.2 <= resize_factor < 1.0:
            ref_small = cv2.resize(ref, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
            alvo_small = cv2.resize(alvo, (0, 0), fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
            scale = 1.0 / resize_factor
        else:
            ref_small = ref
            alvo_small = alvo
            scale = 1.0

        meth = _TM_MAP.get(method_name, cv2.TM_CCOEFF_NORMED)

        # Normalize to 8-bit for matching only (warp is done on original image)
        ref_u8 = cv2.normalize(ref_small, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        alvo_u8 = cv2.normalize(alvo_small, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        res = cv2.matchTemplate(alvo_u8, ref_u8, meth)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if meth in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            top_left = min_loc
        else:
            top_left = max_loc

        dx_small, dy_small = top_left  # template position inside alvo_small
        # We must translate alvo by (-dx, -dy) to align it to ref
        dx = int(round(dx_small * scale))
        dy = int(round(dy_small * scale))
        M = np.array([[1, 0, -dx], [0, 1, -dy]], dtype=np.float32)
        return _warp_with_affine(alvo, M, ref.shape, interp=warp_interp, border=border_mode)

    except Exception:
        return alvo


# -----------------------------
# Mutual information (placeholder)
# -----------------------------
def alinhar_por_mi(ref, alvo, params=None, **kwargs):
    """
    Placeholder mutual-information alignment.

    Currently this method simply delegates to `alinhar_por_template` with any
    parameters provided. A true MI-based alignment would require an optimizer
    (e.g. ITK/SimpleITK) iterating over translation/rotation/scale to maximize
    mutual information between `ref` and `alvo`.

    Parameters
    ----------
    ref : np.ndarray
        Reference image.
    alvo : np.ndarray
        Image to be aligned.
    params : dict, optional
        Parameters passed through to `alinhar_por_template`.

    Returns
    -------
    np.ndarray
        Output of `alinhar_por_template`.
    """
    return alinhar_por_template(ref, alvo, params=params, **kwargs)


# -----------------------------
# SuperGlue (deep matching)
# -----------------------------
def alinhar_por_superglue(ref, alvo, params=None, **kwargs):
    """
    Align the moving image (`alvo`) to the reference (`ref`) using the
    external SuperGlue (deep matching) implementation.

    Parameters
    ----------
    ref : np.ndarray
        Reference image.
    alvo : np.ndarray
        Image to be aligned.
    params : dict, optional
        Parameters are forwarded to the underlying `superglue_match`.

    Returns
    -------
    np.ndarray
        Warped version of `alvo` aligned to `ref`, or the original `alvo`
        if the backend fails.
    """
    return superglue_match(ref, alvo, params=params)


# -----------------------------
# Router
# -----------------------------
def alinhar_imagem(ref, alvo, metodo="orb", params=None, **kwargs):
    """
    High-level alignment router.

    This function dispatches to the chosen alignment backend based on the
    string identifier `metodo`. The identifiers are stable and must not be
    renamed, as they are used across the GUI and configuration files.

    Parameters
    ----------
    ref : np.ndarray
        Reference image (target geometry).
    alvo : np.ndarray
        Image to be aligned.
    metodo : str, optional
        Alignment method identifier. Supported values:
            - "orb"       → ORB keypoints
            - "sift"      → SIFT keypoints
            - "akaze"     → AKAZE keypoints
            - "ecc"       → ECC (intensity-based)
            - "template"  → Template matching
            - "mi"        → Mutual information (placeholder)
            - "superglue" → SuperGlue (deep matching)
        Defaults to "orb".
    params : dict, optional
        Dictionary of parameters forwarded to the chosen backend.
    **kwargs :
        Extra keyword arguments forwarded to the backend.

    Returns
    -------
    np.ndarray
        Warped version of `alvo` aligned to `ref`, or the original `alvo`
        if alignment fails or if the method falls back.
    """
    metodos = {
        "orb": alinhar_por_orb,
        "sift": alinhar_por_sift,
        "akaze": alinhar_por_akaze,
        "ecc": alinhar_por_ecc,
        "template": alinhar_por_template,
        "mi": alinhar_por_mi,
        "superglue": alinhar_por_superglue,
    }
    func = metodos.get(metodo, alinhar_por_orb)
    return func(ref, alvo, params=params, **kwargs)







