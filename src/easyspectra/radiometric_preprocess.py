# radiometric_preprocess.py
# EasySpectra — Radiometric preprocessing (standalone)
#
# Goals:
# - Apply radiometric corrections BEFORE orthomosaicking / cube generation.
# - Two methods:
#     (1) Flight metadata recorded during capture (irradiance/DLS-based; Pix4D-like normalization)
#     (2) Calibration panels (panel-based scaling)
#
# Notes:
# - This module is designed to handle folders with MANY single-band TIFFs (5…hundreds).
# - It does NOT create orthomosaics; it only outputs corrected single-band TIFFs.
# - Keep this module sensor-agnostic: it selects the best available metadata fields rather than hardcoding vendors.
#
# Dependencies: numpy, rasterio, tifffile (already used in geo_import)

from __future__ import annotations

import os
import re
import json
import ast
import math
import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Tuple, List

import numpy as np

# Clamp ratio for flight-metadata normalization to avoid extreme outliers saturating outputs
RATIO_CLAMP_MIN = 0.25
RATIO_CLAMP_MAX = 4.0

try:
    import rasterio
    from rasterio.enums import Resampling
except Exception:  # pragma: no cover
    rasterio = None

import tifffile
import xml.etree.ElementTree as ET



# Keys commonly used by vendors for wavelength / band identification
_WL_KEYS = (
    "WAVELENGTH","wavelength",
    "CENTER_WAVELENGTH","center_wavelength",
    "CENTRAL_WAVELENGTH","central_wavelength",
    "CentralWavelength",
    "Xmp.Camera.CentralWavelength",
    "BandName","BANDNAME","BAND_NAME","band_name",
    "Xmp.Camera.BandName",
    "DESCRIPTION","description",
)
# -----------------------------
# Metadata extraction (copied from geo_import.py, with minimal safe fixes)
# -----------------------------

def scan_tiff_metadata(tif_path: str) -> dict:
    """Scan a TIFF/GeoTIFF and return metadata useful for band labeling and radiometric corrections.

    Returns
    -------
    dict
        Keys (when available):
        - camera_model (str|None)
        - band_name (str|None)
        - central_wavelength_nm (float|None)
        - fwhm_nm (float|None)
        - rig_camera_index (int|None)
        - wavelength_list_nm (list[float]|None)  # for multi-band hyperspectral
        - dls (dict[str, float])  # downwelling light sensor / irradiance fields
        - raw_tags (dict)  # merged tags for debugging
    """
    meta = {
        "camera_model": None,
        "band_name": None,
        "central_wavelength_nm": None,
        "fwhm_nm": None,
        "rig_camera_index": None,
        "wavelength_list_nm": None,
        "dls": {},
        "raw_tags": {},
        "meta_flat": {},
    }

    def _norm(v):
        try:
            if isinstance(v, bytes):
                v = v.decode("utf-8", "ignore")
            return str(v).strip()
        except Exception:
            return None

    def _norm_key(k: object) -> str:
        """Normalize metadata keys for fuzzy matching.
        - lowercase
        - remove spaces and separators: . _ - :
        """
        if k is None:
            return ""
        s = str(k).strip().lower()
        s = re.sub(r"[\s\._\-:]+", "", s)
        return s

    def _parse_float_list(val: object) -> List[float]:
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            out: List[float] = []
            for it in val:
                try:
                    out.append(float(it))
                except Exception:
                    # try extracting from string
                    m = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(it))
                    if m:
                        try:
                            out.append(float(m[0]))
                        except Exception:
                            pass
            return out
        s = str(val)
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        out: List[float] = []
        for n in nums:
            try:
                out.append(float(n))
            except Exception:
                pass
        return out

    def _add_meta_flat(key: str, value: object):
        """Add a value to meta_flat using multiple normalized aliases."""
        if not key:
            return
        nfull = _norm_key(key)
        if nfull:
            meta["meta_flat"][nfull] = value
        # Also add local-name alias (last segment after ':' or '/' or '#')
        local = str(key)
        if ":" in local:
            local = local.split(":")[-1]
        if "/" in local:
            local = local.split("/")[-1]
        if "#" in local:
            local = local.split("#")[-1]
        nloc = _norm_key(local)
        if nloc and nloc not in meta["meta_flat"]:
            meta["meta_flat"][nloc] = value

    def _parse_first_float(s: str):
        if s is None:
            return None
        try:
            return float(s)
        except Exception:
            pass
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    def _parse_first_int(s: str):
        f = _parse_first_float(s)
        if f is None:
            return None
        try:
            return int(round(f))
        except Exception:
            return None

    def _ns_to_prefix(ns: str) -> str:
        """Map known XMP namespaces to stable prefixes (Pix4D-style)."""
        ns_l = (ns or "").lower()
        if "pix4d.com/camera" in ns_l:
            return "Camera"
        if "pix4d.com/dls" in ns_l:
            return "DLS"
        # Many sensors (e.g., MicaSense) embed DLS fields under micasense.com/DLS/...
        if "dls" in ns_l and "micasense" in ns_l:
            return "DLS"
        if "micasense.com/dls" in ns_l:
            return "DLS"
        if "micasense" in ns_l:
            return "MicaSense"
        if "dji" in ns_l:
            return "DJI"
        if "parrot" in ns_l:
            return "Parrot"
        return "XMP"

    def _parse_xmp_packet(xmp_text: str) -> dict:
        """Parse an XMP packet into a flat dict key->text.

        Keys are '<Prefix>:<LocalName>' so different vendors can still be mined
        with heuristics (e.g., 'Camera:CentralWavelength', 'DLS:SpectralIrradiance').
        """
        if not xmp_text:
            return {}
        s = xmp_text
        start = s.find("<x:xmpmeta")
        end = s.rfind("</x:xmpmeta>")
        if start != -1 and end != -1:
            s = s[start : end + len("</x:xmpmeta>")]
        try:
            root = ET.fromstring(s)
        except Exception:
            return {}

        out = {}
        for elem in root.iter():
            tag = elem.tag
            if isinstance(tag, str) and tag.startswith("{") and "}" in tag:
                ns, local = tag[1:].split("}", 1)
                prefix = _ns_to_prefix(ns)
                key = f"{prefix}:{local}"
            else:
                key = str(tag)

            txt = (elem.text or "").strip()
            if txt:
                out[key] = txt

            for ak, av in (elem.attrib or {}).items():
                if av:
                    out[str(ak)] = str(av).strip()
        return out

    # ---- 2a) Rasterio tags/description (when available) ----
    if rasterio is not None:
        try:
            with rasterio.open(tif_path) as src:
                try:
                    tags = src.tags() or {}
                    for k, v in tags.items():
                        meta["raw_tags"][f"RI:{k}"] = str(v)
                        _add_meta_flat(k, v)
                        _add_meta_flat(f"RI:{k}", v)
                except Exception:
                    pass
                try:
                    desc = src.descriptions[0] if src.descriptions else None
                    if desc:
                        meta["raw_tags"]["RI:description"] = str(desc)
                        _add_meta_flat("description", desc)
                        _add_meta_flat("RI:description", desc)
                except Exception:
                    pass
        except Exception as e:
            meta["raw_tags"]["_rasterio_error"] = repr(e)

    # ---- 2b) TIFF EXIF/GPS/XMP blocks via tifffile ----
    try:
        with tifffile.TiffFile(tif_path) as tif:
            page = tif.pages[0]
            ttags = page.tags

            def _tag_value(name_or_code):
                try:
                    return ttags[name_or_code].value
                except Exception:
                    return None

            # Make/Model
            for k in ("Make", "Model"):
                v = _tag_value(k)
                if v:
                    meta["camera_model"] = meta["camera_model"] or _norm(v)

            # Bits per sample (TIFF tag)
            try:
                bps = _tag_value("BitsPerSample")
                if bps is None:
                    bps = _tag_value(258)
                if bps is not None:
                    meta["raw_tags"]["TIFF:BitsPerSample"] = str(bps)
                    _add_meta_flat("BitsPerSample", bps)
            except Exception:
                pass

            # Black level / dark offset (TIFF tag; may be scalar or tuple)
            try:
                bl = _tag_value("BlackLevel")
                if bl is None:
                    bl = _tag_value(50714)  # common GeoTIFF code used by some writers
                if bl is not None:
                    meta["raw_tags"]["TIFF:BlackLevel"] = str(bl)
                    _add_meta_flat("BlackLevel", bl)
                    # Also store a parsed list for convenience
                    bl_list = _parse_float_list(bl)
                    if bl_list:
                        meta["raw_tags"]["_blacklevel_list"] = bl_list
                        _add_meta_flat("BlackLevelList", bl_list)
            except Exception:
                pass

            # XMP (tag 700)
            xmp_text = None
            xv = _tag_value("XMP")
            if xv:
                if isinstance(xv, (bytes, bytearray)):
                    xmp_text = xv.decode("utf-8", "ignore")
                else:
                    xmp_text = str(xv)

            if xmp_text:
                xmp = _parse_xmp_packet(xmp_text)
                for k, v in xmp.items():
                    meta["raw_tags"][f"XMP:{k}"] = v
                    _add_meta_flat(k, v)
                    _add_meta_flat(f"XMP:{k}", v)

                rig = xmp.get("Camera:RigCameraIndex") or xmp.get("MicaSense:RigCameraIndex")
                if rig is not None:
                    meta["rig_camera_index"] = _parse_first_int(rig)

                bn = xmp.get("Camera:BandName") or xmp.get("MicaSense:BandName")
                if bn:
                    meta["band_name"] = _norm(bn)

                cw = xmp.get("Camera:CentralWavelength") or xmp.get("MicaSense:CentralWavelength")
                if cw is not None:
                    meta["central_wavelength_nm"] = _parse_first_float(cw)

                fwhm = xmp.get("Camera:WavelengthFWHM") or xmp.get("MicaSense:WavelengthFWHM")
                if fwhm is not None:
                    meta["fwhm_nm"] = _parse_first_float(fwhm)

                # Collect irradiance-like fields for flight-metadata corrections
                for k, v in xmp.items():
                    kl = (k or "").lower()
                    if ("irradi" in kl) or ("downwelling" in kl) or ("spectralirradiance" in kl):
                        fv = _parse_first_float(v)
                        if fv is not None:
                            meta["dls"][k] = fv
    except Exception as e:
        meta["raw_tags"]["_tifffile_error"] = repr(e)

    # ---- Band name fallback (from raw tags) ----
    if not meta["band_name"]:
        for k in ("RI:description", "RI:BAND_NAME", "RI:BandName"):
            if k in meta["raw_tags"]:
                meta["band_name"] = _norm(meta["raw_tags"][k])
                break

    # ---- Central wavelength fallback (from raw tags) ----
    if meta["central_wavelength_nm"] is None:
        for k in ("XMP:Camera:CentralWavelength", "XMP:MicaSense:CentralWavelength"):
            v = meta["raw_tags"].get(k)
            if v:
                meta["central_wavelength_nm"] = _parse_first_float(v)
                break


    # ---- Central wavelength fallback (from rasterio tags/descriptions) ----
    if meta["central_wavelength_nm"] is None:
        # Try description first (often contains '475' etc.)
        desc = meta["raw_tags"].get("RI:description")
        nm = _parse_first_float(desc) if desc else None
        if nm is not None and 300 <= nm <= 2500:
            meta["central_wavelength_nm"] = float(nm)
        else:
            # Try known wavelength keys in rasterio tags
            for k in _WL_KEYS:
                v = meta["raw_tags"].get(f"RI:{k}")
                if v:
                    nm = _parse_first_float(v)
                    if nm is not None and 300 <= nm <= 2500:
                        meta["central_wavelength_nm"] = float(nm)
                        break

    # ---- Sidecar wavelength fallback (.dat/.xmp/.xml/.json/.txt) ----
    if meta["central_wavelength_nm"] is None:
        base, _ = os.path.splitext(tif_path)
        sidecars = [base + ext for ext in (".dat",".xmp",".xml",".json",".txt")]
        for sp in sidecars:
            if not os.path.exists(sp):
                continue
            try:
                with open(sp, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
            except Exception:
                continue
            # simple key/value scans
            # 1) try key-like patterns
            nm_found = None
            for k in _WL_KEYS:
                # e.g., CentralWavelength=475 or <CentralWavelength>475</...>
                mm = re.search(rf"{re.escape(k)}\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", txt)
                if mm:
                    nm_found = _parse_first_float(mm.group(1))
                    break
                mm = re.search(rf"<{re.escape(k)}[^>]*>\s*([0-9]+(?:\.[0-9]+)?)\s*</", txt)
                if mm:
                    nm_found = _parse_first_float(mm.group(1))
                    break
            # 2) fallback: first plausible nm number in file
            if nm_found is None:
                nm_found = _parse_first_float(txt)
            if nm_found is not None and 300 <= nm_found <= 2500:
                meta["central_wavelength_nm"] = float(nm_found)
                break

    return meta


# -----------------------------
# Radiometric correction core
# -----------------------------

@dataclass
class RadiometricConfig:
    """
    Configuration parameters for radiometric preprocessing.

    Attributes
    ----------
    input_folder : str
        Path to the folder containing single-band TIFFs.
    method : str
        Which correction method to use. One of:
          * ``"flight_metadata"`` – legacy mode using only irradiance tags and
            optional sun-angle compensation.
          * ``"calibration_panels"`` – use a calibrated reflectance panel to
            compute a scale factor per band.
          * ``"auto"`` – automatically choose between multiple radiometric
            correction strategies (see Recommendation A). The most complete
            strategy available for each image will be selected at run-time.
    output_folder : Optional[str], default None
        Destination folder for corrected GeoTIFFs. If None, a subfolder is
        created inside ``input_folder``.
    apply_sun_angle : bool, default True
        Whether to multiply irradiance by the sine of the solar elevation angle
        when normalising by flight metadata (only relevant for methods that
        reference DLS measurements).
    panel_reflectance : float, default 0.60
        User-provided reflectance value of the calibration panel (range 0–1).
    panel_filename_hint : str, default "panel"
        Substring used to detect panel images when using the calibration_panels
        method.
    recursive : bool, default False
        Search for TIFFs recursively under ``input_folder``.
    verbose : bool, default False
        If True, print progress information.
    panel_roi : Optional[Tuple[int, int, int, int]], default None
        Optional manual region-of-interest for panel images specified as
        (x, y, width, height). If None, a centered 20 % ROI is used.
    overwrite : bool, default False
        If False, skip writing output files that already exist.
    output_dtype : str, default "uint16"
        Pixel type for GeoTIFF output. One of ``"float32"`` or ``"uint16"``.
        Note that metadata preservation requires the target TIFF to retain its
        original data type. As a result, float outputs are most safely
    uint16_scale : Optional[float], default None
        Scale factor applied before casting reflectance to ``uint16``. If
        ``None``, the default depends on ``method``: 1 for
        ``flight_metadata``, 10 000 for ``calibration_panels``, and 65 535
        for ``auto`` (reflectance scaled to full 16‑bit range).
    keep_panel_images : bool, default False
        Whether to also write corrected panel images when using the
        calibration_panels method.
    """

    input_folder: str
    method: str
    output_folder: Optional[str] = None
    apply_sun_angle: bool = True
    panel_reflectance: float = 0.60
    panel_filename_hint: str = "panel"
    recursive: bool = False
    verbose: bool = False
    panel_roi: Optional[Tuple[int, int, int, int]] = None
    overwrite: bool = False
    output_dtype: str = "uint16"
    uint16_scale: Optional[float] = None
    keep_panel_images: bool = False

    # --- Panel-based calibration inputs (calibration_panels) ---
    panel_folder: Optional[str] = None  # folder containing panel TIFFs (can differ from input_folder)
    panel_image_paths: Optional[List[str]] = None  # explicit list of panel TIFF paths
    # ROI per band-key (e.g., "wl_475" or "rig_0" or "band_red") -> (x,y,w,h)
    panel_roi_by_band: Optional[Dict[str, Tuple[int, int, int, int]]] = None
    # Reflectance per band-key (0..1). If missing for a band, falls back to panel_reflectance.
    panel_reflectance_by_band: Optional[Dict[str, float]] = None

    # --- Optional: panels captured BEFORE and AFTER the flight (calibration_panels) ---
    # If enabled, compute scale_before and scale_after per band and interpolate by capture time.
    panel_before_after: bool = False
    panel_folder_before: Optional[str] = None
    panel_folder_after: Optional[str] = None
    panel_image_paths_before: Optional[List[str]] = None
    panel_image_paths_after: Optional[List[str]] = None
    panel_roi_by_band_before: Optional[Dict[str, Tuple[int, int, int, int]]] = None
    panel_roi_by_band_after: Optional[Dict[str, Tuple[int, int, int, int]]] = None
    # Robust extraction options to handle 'outsides' around the panel
    panel_roi_inner_frac: float = 0.80  # use central fraction of ROI to avoid borders/outside objects
    panel_dn_percentiles: Tuple[float, float] = (10.0, 90.0)  # trim extremes within ROI
    # --- Robust shielding for auto/irradiance (no panel) ---
    robust_irradiance: bool = True
    robust_window: int = 7  # rolling median window (odd recommended)
    robust_mad_k: float = 6.0  # MAD multiplier for robust limits

    # --- Optional physical correction: Band sensitivity (if available in metadata) ---
    # "divide" (default) -> reflectance /= sensitivity
    # "multiply"         -> reflectance *= sensitivity
    # "off"              -> never apply even if present
    band_sensitivity_mode: str = "divide"
def _list_tiffs(folder: str, recursive: bool = False) -> List[str]:
    exts = (".tif", ".tiff", ".TIF", ".TIFF")
    out: List[str] = []
    if recursive:
        for root, _, files in os.walk(folder):
            for name in files:
                if name.endswith(exts) or name.lower().endswith((".tif", ".tiff")):
                    out.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            if name.endswith(exts) or name.lower().endswith((".tif", ".tiff")):
                out.append(os.path.join(folder, name))
    out.sort()
    return out


def _safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Helpers (band keys, time ordering, robust stats)
# -----------------------------

def _band_key_from_meta(meta: dict) -> str:
    """Return a stable band key for grouping files in pre-passes."""
    try:
        if meta.get("central_wavelength_nm") is not None:
            return f"wl_{int(round(float(meta['central_wavelength_nm'])))}"
    except Exception:
        pass
    bn = meta.get("band_name")
    if bn:
        return f"band_{str(bn).strip().lower()}"
    rig = meta.get("rig_camera_index")
    if rig is not None:
        return f"rig_{int(rig)}"
    return "band_unknown"


def _get_capture_time_key(path: str) -> float:
    """Best-effort capture-time key for sorting (seconds since epoch).

    Falls back to file mtime if TIFF DateTime is missing/unparseable.
    """
    # TIFF DateTime tag is 306 (string 'YYYY:MM:DD HH:MM:SS')
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            dt = None
            try:
                dt = page.tags.get("DateTime").value
            except Exception:
                try:
                    dt = page.tags.get(306).value
                except Exception:
                    dt = None
            if dt:
                if isinstance(dt, bytes):
                    dt = dt.decode("utf-8", "ignore")
                s = str(dt).strip()
                # Some cameras omit seconds; tolerate that
                for fmt in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d %H:%M"):
                    try:
                        t = datetime.datetime.strptime(s, fmt)
                        return float(t.timestamp())
                    except Exception:
                        pass
    except Exception:
        pass
    try:
        return float(os.path.getmtime(path))
    except Exception:
        return 0.0


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median with edge handling."""
    v = np.asarray(values, dtype=np.float64)
    n = int(v.size)
    if n == 0:
        return v
    w = int(window) if window and window > 0 else 1
    if w % 2 == 0:
        w += 1
    half = w // 2
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = v[lo:hi]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(np.median(seg))
    return out


def _mad(x: np.ndarray, center: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    d = np.abs(x - float(center))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    return float(np.median(d))


def _compute_irradiance_gain(meta: dict, irr_ref: Optional[float], cfg: RadiometricConfig) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], bool]:
    """Compute the multiplicative gain applied in the irradiance strategy.

    Returns
    -------
    g_raw : float|None
        Gain factor applied to arr_f (before bit-depth scaling).
    denom : float|None
        Denominator used (irradiance * optional sun-angle term).
    irr_raw : float|None
        Raw irradiance field chosen from metadata.
    solar_elev : float|None
        Solar elevation degrees (if detected).
    has_ref : bool
        Whether irr_ref was used (True) or the strategy fell back to 1/denom (False).
    """
    irr_raw = _pick_irradiance(meta)
    if irr_raw is None or irr_raw <= 0:
        return None, None, None, None, False
    denom = float(irr_raw)
    solar_elev = _pick_solar_elevation_deg(meta)
    if cfg.apply_sun_angle and solar_elev is not None:
        denom *= max(1e-6, math.sin(math.radians(float(solar_elev))))
    if denom <= 0:
        return None, denom, float(irr_raw), solar_elev, bool(irr_ref)
    if irr_ref and irr_ref > 0:
        g_raw = float(irr_ref) / denom
        return g_raw, denom, float(irr_raw), solar_elev, True
    # no ref: gain is 1/denom (previous behaviour)
    g_raw = 1.0 / max(1e-12, denom)
    return float(g_raw), denom, float(irr_raw), solar_elev, False


def _pick_irradiance(meta: dict) -> Optional[float]:
    """Pick the best irradiance value from scan_tiff_metadata output."""
    dls: Dict[str, float] = meta.get("dls") or {}
    if not dls:
        return None
    # Priority list (common Pix4D/MicaSense patterns)
    priority = [
        "Camera:Irradiance",
        "DLS:SpectralIrradiance",
        "MicaSense:SpectralIrradiance",
        "DLS:HorizontalIrradiance",
        "MicaSense:HorizontalIrradiance",
        "DLS:DirectIrradiance",
        "MicaSense:DirectIrradiance",
    ]
    for k in priority:
        if k in dls:
            return float(dls[k])
    # fallback: any irradiance-like field
    for k, v in dls.items():
        if v is None:
            continue
        if "irradi" in k.lower():
            return float(v)
    return None




def _pick_band_sensitivity(meta: dict) -> Optional[float]:
    """Pick a band sensitivity coefficient if present (sensor-agnostic).

    We search the unified meta_flat first (normalized keys), then fall back to raw_tags.
    Accepts scalar or list/tuple. Returns a single float (first value) if found.
    """
    # Prefer meta_flat normalized keys
    mf = meta.get("meta_flat") or {}
    for key_norm in ("bandsensitivity", "band_sensitivity", "spectralsensitivity", "sensitivity"):
        kn = re.sub(r"[\s\._\-:]+", "", str(key_norm).lower())
        if kn in mf:
            val = mf.get(kn)
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(val))
            if nums:
                try:
                    return float(nums[0])
                except Exception:
                    pass
            if isinstance(val, (list, tuple)) and len(val) > 0:
                try:
                    return float(val[0])
                except Exception:
                    pass

    raw = meta.get("raw_tags") or {}
    # Heuristic: look for any tag containing 'bandsensit'
    for rk, rv in raw.items():
        if not rk:
            continue
        if "bandsensit" in str(rk).lower().replace("_","").replace("-",""):
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(rv))
            if nums:
                try:
                    return float(nums[0])
                except Exception:
                    continue
    return None

def _pick_solar_elevation_deg(meta: dict) -> Optional[float]:
    """Pick solar elevation (degrees) if present.

    Some sensors store angles in radians. Heuristic:
    - If 0 < value <= 3.2 -> assume radians and convert to degrees.
    """
    raw = meta.get("raw_tags") or {}
    candidates = [
        raw.get("XMP:DLS:SolarElevation"),
        raw.get("XMP:MicaSense:SolarElevation"),
        raw.get("XMP:Camera:SolarElevation"),
    ]
    for c in candidates:
        if c is None:
            continue
        try:
            val = float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(c))[0])
        except Exception:
            continue
        if 0 < val <= 3.2:
            val = math.degrees(val)
        return float(val)
    return None


def _read_tiff_array_and_profile(path: str):
    """Read single-band TIFF to array and a rasterio-like profile for writing."""
    if rasterio is not None:
        with rasterio.open(path) as src:
            arr = src.read(1)
            profile = src.profile.copy()
            tags = src.tags()
        return arr, profile, tags
    # fallback (no georeferencing preservation)
    arr = tifffile.imread(path)
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": str(arr.dtype),
        "crs": None,
        "transform": None,
    }
    tags = {}
    return arr, profile, tags


# -----------------------------------------------------------------------------
# Additional radiometric parameter extraction
#
# The following helper extracts EXIF/XMP parameters required for the
# high‑fidelity radiometric calibration (Recommendation A).  It is
# intentionally conservative: if a parameter is missing, it is returned as
# ``None`` rather than assuming a default.  This allows the caller to
# gracefully fall back to simpler strategies.

def _extract_exif_params(path: str) -> Dict[str, Optional[float | Tuple[float, ...]]]:
    """Extract radiometric EXIF/XMP parameters from a single‑band TIFF.

    Parameters
    ----------
    path : str
        Path to the TIFF file.

    Returns
    -------
    dict
        A dictionary with the following optional keys. Any missing key will
        have a value of ``None``:

        ``exposure`` : float
            Exposure time in seconds.
        ``iso`` : float
            ISO speed as a numeric gain (ISO value divided by 100). For
            example, an ISO tag of 200 will be returned as 2.0.
        ``black_level`` : float
            Mean black level offset (digital number) across channels.
        ``bits`` : int
            Number of bits per pixel (e.g. 12 or 16).
        ``radiometric_calibration`` : Tuple[float, float, float]
            Radiometric calibration coefficients (ak1, ak2, ak3) if present.
    """
    params: Dict[str, Optional[float | Tuple[float, ...]]] = {
        "exposure": None,
        "iso": None,
        "black_level": None,
        "bits": None,
        "radiometric_calibration": None,
    }
    try:
        with tifffile.TiffFile(path) as tif:
            page = tif.pages[0]
            tags = page.tags

            # Exposure time (Exif tag 33434)
            try:
                # Some TIFFs store ExposureTime directly as a float or ratio tuple
                exp_tag = tags.get("ExposureTime") or tags.get(33434)
                if exp_tag is not None:
                    val = exp_tag.value
                    if isinstance(val, tuple) and len(val) == 2:
                        # convert rational (num, denom)
                        params["exposure"] = float(val[0]) / float(val[1]) if float(val[1]) != 0 else None
                    else:
                        params["exposure"] = float(val)
            except Exception:
                pass

            # ISO speed (Exif tag 34855 or sometimes stored as ISOSpeed)
            iso_val: Optional[float] = None
            try:
                iso_tag = tags.get("ISOSpeedRatings") or tags.get("ISOSpeed") or tags.get(34855)
                if iso_tag is not None:
                    iso_raw = iso_tag.value
                    if isinstance(iso_raw, (tuple, list)):
                        iso_val = float(iso_raw[0])
                    else:
                        iso_val = float(iso_raw)
            except Exception:
                pass
            if iso_val and iso_val > 0:
                params["iso"] = iso_val / 100.0

            # Bits per sample (generally a tuple of ints per channel)
            try:
                bps_tag = tags.get("BitsPerSample") or tags.get(258)
                if bps_tag is not None:
                    bps = bps_tag.value
                    if isinstance(bps, (tuple, list)):
                        params["bits"] = int(bps[0])
                    else:
                        params["bits"] = int(bps)
            except Exception:
                pass

            # Black level offset (Exif tag 50714) – returns array of four values
            try:
                bl_tag = tags.get("BlackLevel") or tags.get(50714)
                if bl_tag is not None:
                    bl_val = bl_tag.value
                    # Some cameras store four values; average them
                    if isinstance(bl_val, (tuple, list)):
                        bl_arr = [float(x) for x in bl_val]
                        params["black_level"] = float(sum(bl_arr) / len(bl_arr))
                    else:
                        params["black_level"] = float(bl_val)
            except Exception:
                pass

            # Radiometric calibration coefficients stored in XMP under
            # MicaSense:RadiometricCalibration or Camera:RadiometricCalibration.
            # We'll attempt to parse the XMP packet similar to scan_tiff_metadata.
            xmp_text = None
            try:
                xmp_tag = tags.get("XMP") or tags.get(700)
                if xmp_tag is not None:
                    xv = xmp_tag.value
                    if isinstance(xv, (bytes, bytearray)):
                        xmp_text = xv.decode("utf-8", "ignore")
                    else:
                        xmp_text = str(xv)
            except Exception:
                pass
            if xmp_text:
                xmp = _parse_xmp_packet(xmp_text)
                calib_val = None
                for key in (
                    "MicaSense:RadiometricCalibration",
                    "Camera:RadiometricCalibration",
                    "RadiometricCalibration",
                ):
                    if key in xmp:
                        calib_val = xmp[key]
                        break
                if calib_val:
                    try:
                        parts = [float(x.strip()) for x in str(calib_val).replace(";", ",").split(",") if x.strip()]
                        if len(parts) >= 1:
                            # Only the first coefficient is commonly used; others are row gradient and not used here.
                            # We still return all coefficients found for completeness.
                            params["radiometric_calibration"] = tuple(parts)
                    except Exception:
                        pass
    except Exception:
        # Unable to parse some EXIF tags; leave defaults as None
        pass
    return params



def _write_tiff(path: str, arr: np.ndarray, profile: dict, tags: dict, src_path: str | None = None) -> None:
    """
    Write corrected pixels while *preserving* all metadata from the original TIFF.

    Requirement (photogrammetry-friendly):
    - ALWAYS preserve TIFFTAG/EXIF/XMP/GeoTIFF blocks by copying the original file
      byte-for-byte and only overwriting raster samples in-place.

    Notes
    -----
    - This keeps: TIFF layout, compression, tiling/strip layout, interleave,
      band count, GeoTIFF keys, EXIF/XMP maker notes, etc. (as long as we do not
      recreate the file).
    - We only fall back to writing a brand-new TIFF if src_path is not provided
      or cannot be copied/opened.
    """
    if rasterio is None:
        # Without rasterio, we cannot safely update in-place.
        # To respect the "preserve metadata ALWAYS" requirement, we refuse.
        raise RuntimeError("rasterio is required to preserve TIFF metadata by in-place pixel overwrite.")

    import shutil

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if (not src_path) or (not os.path.exists(src_path)):
        raise FileNotFoundError(f"src_path not found: {src_path!r}. Cannot preserve TIFF metadata without the original file.")

    # Always start from a fresh copy of the original so we never accumulate partial writes.
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass
    shutil.copy2(src_path, path)

    # Overwrite samples in-place (no profile changes = metadata preserved).
    with rasterio.open(path, "r+") as dst:
        dst_dtype = np.dtype(dst.dtypes[0])

        # --- shape / band handling ---
        # rasterio expects (count, height, width)
        write_arr = arr

        if write_arr.ndim == 2:
            # single band (H, W)
            if dst.count != 1:
                raise ValueError(f"Output TIFF has {dst.count} bands but array is 2D.")
            write_arr = write_arr[np.newaxis, :, :]
        elif write_arr.ndim == 3:
            # either (bands, H, W) or (H, W, bands)
            if write_arr.shape[0] == dst.count and write_arr.shape[1] == dst.height and write_arr.shape[2] == dst.width:
                pass  # already (bands, H, W)
            elif write_arr.shape[2] == dst.count and write_arr.shape[0] == dst.height and write_arr.shape[1] == dst.width:
                write_arr = np.transpose(write_arr, (2, 0, 1))  # (H, W, bands) -> (bands, H, W)
            else:
                raise ValueError(
                    f"Array shape {write_arr.shape} does not match TIFF (count={dst.count}, height={dst.height}, width={dst.width})."
                )
        else:
            raise ValueError(f"Unsupported array ndim={write_arr.ndim}; expected 2D or 3D.")

        # --- dtype handling ---
        # We strongly prefer writing in the original dtype (typically UInt16 for MicaSense).
        if np.dtype(write_arr.dtype) != dst_dtype:
            if np.issubdtype(dst_dtype, np.integer):
                info = np.iinfo(dst_dtype)
                # If float-like, optionally scale 0..1 -> full integer range.
                wa = write_arr.astype(np.float32, copy=False)

                # Heuristic: if values look like reflectance (0..1-ish), scale to full range.
                finite = wa[np.isfinite(wa)]
                vmax = float(np.nanmax(finite)) if finite.size else 0.0
                vmin = float(np.nanmin(finite)) if finite.size else 0.0

                if vmax <= 1.5 and vmin >= -0.1:
                    wa = wa * float(info.max)

                wa = np.nan_to_num(wa, nan=0.0, posinf=float(info.max), neginf=0.0)
                wa = np.rint(wa)
                wa = np.clip(wa, info.min, info.max).astype(dst_dtype)
                write_arr = wa
            else:
                write_arr = write_arr.astype(dst_dtype)

        # final sanity check
        if write_arr.shape != (dst.count, dst.height, dst.width):
            raise ValueError(f"Final write array shape {write_arr.shape} != {(dst.count, dst.height, dst.width)}")

        dst.write(write_arr)

        # Do NOT wipe existing tags; only optionally add extra default-namespace tags.
        if tags:
            try:
                dst.update_tags(**tags)
            except Exception:
                # Non-fatal; tags are already preserved from the copied file.
                pass


def _to_output_dtype(arr: np.ndarray, cfg: RadiometricConfig) -> np.ndarray:
    if cfg.output_dtype.lower() == "uint16":
        scaled = np.clip(arr * float(cfg.uint16_scale), 0, 65535)
        return scaled.astype(np.uint16)
    return arr.astype(np.float32, copy=False)


def _default_panel_roi(arr: np.ndarray) -> Tuple[int, int, int, int]:
    """Fallback ROI: centered box (20% of width/height)."""
    h, w = arr.shape[:2]
    rw = max(10, int(w * 0.2))
    rh = max(10, int(h * 0.2))
    x = int((w - rw) / 2)
    y = int((h - rh) / 2)
    return (x, y, rw, rh)


def _norm_band_key(key: object) -> str:
    """Normalize a user-provided band identifier into our internal band_key."""
    if key is None:
        return "band_unknown"
    # numeric wavelength
    if isinstance(key, (int, float)) and not isinstance(key, bool):
        try:
            return f"wl_{int(round(float(key)))}"
        except Exception:
            pass
    s = str(key).strip()
    if not s:
        return "band_unknown"
    s = s.lower()
    # accept raw wavelength strings like "475" or "475nm"
    if re.fullmatch(r"\d{3,4}(?:\.\d+)?", s):
        return f"wl_{int(round(float(s)))}"
    if s.endswith("nm") and s[:-2].strip().isdigit():
        return f"wl_{int(s[:-2].strip())}"
    # already in expected format
    return s


def _panel_dn_from_roi(panel_arr: np.ndarray,
                       roi: Tuple[int, int, int, int],
                       inner_frac: float = 0.80,
                       dn_percentiles: Tuple[float, float] = (10.0, 90.0)) -> float:
    """Robust DN estimate from a panel ROI, tolerant to 'outsides' around the panel."""
    x, y, w, h = roi
    patch = panel_arr[y:y+h, x:x+w].astype(np.float64, copy=False)
    if patch.size == 0:
        return float("nan")

    # Use a central sub-ROI to avoid borders / outside objects
    try:
        inner_frac = float(inner_frac)
    except Exception:
        inner_frac = 0.80
    inner_frac = max(0.10, min(1.0, inner_frac))

    if inner_frac < 1.0:
        ph, pw = patch.shape[:2]
        iw = max(1, int(round(pw * inner_frac)))
        ih = max(1, int(round(ph * inner_frac)))
        x0 = max(0, int((pw - iw) / 2))
        y0 = max(0, int((ph - ih) / 2))
        patch = patch[y0:y0+ih, x0:x0+iw]

    flat = patch.reshape(-1)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return float("nan")

    # Percentile trimming to remove shadows/speculars/foreign objects
    try:
        p_lo, p_hi = dn_percentiles
        p_lo = float(p_lo); p_hi = float(p_hi)
    except Exception:
        p_lo, p_hi = 10.0, 90.0
    p_lo = max(0.0, min(49.0, p_lo))
    p_hi = max(51.0, min(100.0, p_hi))

    lo = np.percentile(flat, p_lo)
    hi = np.percentile(flat, p_hi)
    trimmed = flat[(flat >= lo) & (flat <= hi)]
    if trimmed.size == 0:
        trimmed = flat

    dn = float(np.nanmedian(trimmed))
    return dn


def _compute_panel_scale(panel_arr: np.ndarray,
                         roi: Tuple[int, int, int, int],
                         panel_reflectance: float,
                         inner_frac: float = 0.80,
                         dn_percentiles: Tuple[float, float] = (10.0, 90.0)) -> Tuple[float, float]:
    """Return (scale, dn_used) where reflectance = DN * scale."""
    dn_used = _panel_dn_from_roi(panel_arr, roi, inner_frac=inner_frac, dn_percentiles=dn_percentiles)
    if (not np.isfinite(dn_used)) or dn_used <= 0:
        return 1.0, float(dn_used)
    scale = float(panel_reflectance) / float(dn_used)
    if (not np.isfinite(scale)) or scale <= 0:
        scale = 1.0
    return float(scale), float(dn_used)


# -----------------------------------------------------------------------------
# Radiometric strategies for Recommendation A
#
# These functions implement the three paths described by the user. Each takes
# the raw image array (arr_f), the extracted EXIF params, the metadata dict
# returned by ``scan_tiff_metadata`` and returns either a corrected float32
# reflectance array or ``None`` if the strategy is not applicable.

def _compute_reflectance_full(arr_f: np.ndarray, exif: Dict[str, Optional[float | Tuple[float, ...]]], meta: dict) -> Optional[np.ndarray]:
    """Compute reflectance using the full radiometric model.

    This strategy requires exposure time, ISO, black level, radiometric calibration
    coefficients and both irradiance and solar elevation tags. The
    implementation here follows an approximate form of the MicaSense model and
    intentionally ignores vignette and row‐gradient corrections to keep the
    calculations lightweight. If any required field is missing, ``None`` is
    returned and a less complete strategy should be used.

    Parameters
    ----------
    arr_f : ndarray
        Raw image data converted to float32.
    exif : dict
        Output of ``_extract_exif_params``.
    meta : dict
        Metadata returned by ``scan_tiff_metadata``.

    Returns
    -------
    ndarray or None
        Reflectance image with values ideally in the 0..1 range, or ``None`` if
        full correction cannot be applied.
    """
    # Check required parameters
    exposure = exif.get("exposure")
    iso_gain = exif.get("iso")
    bits = exif.get("bits")
    black_level = exif.get("black_level")
    radiocal = exif.get("radiometric_calibration")
    # Use only the first coefficient (ak1); additional coefficients correct row
    # gradients and are ignored here.
    ak1: Optional[float] = None
    if radiocal:
        try:
            ak1 = float(radiocal[0]) if len(radiocal) > 0 else None
        except Exception:
            ak1 = None
    irr = _pick_irradiance(meta)
    solar_elev = _pick_solar_elevation_deg(meta)
    if not (exposure and iso_gain and bits and black_level is not None and ak1 and irr and solar_elev):
        return None
    # Validate values
    if exposure <= 0 or iso_gain <= 0 or bits <= 0 or irr <= 0:
        return None

    try:
        # Normalise digital numbers to [0, 1]
        dn_max = float(2 ** int(bits))
        # subtract black level, then divide by bit depth
        norm_dn = (arr_f - float(black_level)) / dn_max
        # radiance = (norm_dn * ak1) / (gain * exposure)
        gain = float(iso_gain)
        radiance = (norm_dn * float(ak1)) / (gain * float(exposure))
        # convert radiance to reflectance using downwelling irradiance and solar elevation
        # reflectance = pi * radiance / (irradiance * cos(theta))
        # cos(theta) = sin(elevation)
        cos_term = max(1e-6, math.sin(math.radians(float(solar_elev))))
        reflectance = (math.pi * radiance) / (float(irr) * cos_term)
        # remove negative values
        reflectance = np.clip(reflectance, 0.0, 1.0)
        return reflectance.astype(np.float32, copy=False)
    except Exception:
        return None


def _compute_reflectance_irradiance(arr_f: np.ndarray, exif: Dict[str, Optional[float | Tuple[float, ...]]], meta: dict, irr_ref: Optional[float], cfg: RadiometricConfig) -> Optional[np.ndarray]:
    """Compute reflectance using the intermediate irradiance normalisation.

    This strategy uses whatever DLS irradiance and solar elevation information is
    available to normalise images relative to a flight‐wide reference, similar
    to the existing ``flight_metadata`` method. If bit depth is known, the
    digital numbers are scaled to reflectance by dividing by 2^bits. Exposure
    and ISO values are not required. If insufficient data exists, ``None`` is
    returned.

    Parameters
    ----------
    arr_f : ndarray
        Raw image data converted to float32.
    exif : dict
        Output of ``_extract_exif_params`` (may include ``bits``).
    meta : dict
        Metadata returned by ``scan_tiff_metadata``.
    irr_ref : float or None
        Reference irradiance across the flight (median of valid irradiances).
    cfg : RadiometricConfig
        Config object used to determine whether to apply sun angle.

    Returns
    -------
    ndarray or None
        Reflectance image (0..1), or ``None`` if no irradiance is available.
    """
    irr = _pick_irradiance(meta)
    if irr is None or irr <= 0:
        return None
    denom = float(irr)
    solar_elev = _pick_solar_elevation_deg(meta)
    if cfg.apply_sun_angle and solar_elev is not None:
        denom *= max(1e-6, math.sin(math.radians(float(solar_elev))))
    # If there is a flight reference irradiance, normalise by the ratio
    arr_corr = arr_f.copy()
    if irr_ref and irr_ref > 0:
        ratio = float(irr_ref) / denom
        # clamp extreme ratios as in the original implementation
        if ratio < RATIO_CLAMP_MIN:
            ratio = RATIO_CLAMP_MIN
        elif ratio > RATIO_CLAMP_MAX:
            ratio = RATIO_CLAMP_MAX
        arr_corr = arr_corr * ratio
    else:
        # fallback: divide by irradiance directly
        arr_corr = arr_corr / max(1e-12, denom)

    # Convert to reflectance by dividing by bit depth if known
    bits = exif.get("bits")
    dn_max = float(2 ** int(bits)) if bits else None
    if dn_max:
        reflectance = arr_corr / dn_max
    else:
        # assume 16 bits if unknown
        reflectance = arr_corr / 65535.0
    reflectance = np.clip(reflectance, 0.0, 1.0)
    return reflectance.astype(np.float32, copy=False)


def _compute_reflectance_fallback(arr_f: np.ndarray, exif: Dict[str, Optional[float | Tuple[float, ...]]]) -> np.ndarray:
    """Compute reflectance using a minimal normalisation.

    This strategy is used when there is insufficient radiometric metadata. It
    scales digital numbers into the 0–1 range by dividing by the maximum
    representable value according to the bit depth. If bit depth is unknown it
    assumes 16 bits.

    Parameters
    ----------
    arr_f : ndarray
        Raw image data converted to float32.
    exif : dict
        Output of ``_extract_exif_params`` (may include ``bits``).

    Returns
    -------
    ndarray
        Reflectance image (0..1) computed by normalising to bit depth.
    """
    bits = exif.get("bits")
    dn_max = float(2 ** int(bits)) if bits else 65535.0
    bl = exif.get("black_level")
    if bl is not None:
        try:
            arr_f = arr_f - float(bl)
        except Exception:
            pass
    reflectance = arr_f / dn_max
    reflectance = np.clip(reflectance, 0.0, 1.0)
    return reflectance.astype(np.float32, copy=False)


def apply_radiometric_corrections(cfg: RadiometricConfig) -> dict:
    """Apply radiometric corrections to a folder of single‑band TIFFs.

    The behaviour depends on ``cfg.method``:

      * ``flight_metadata`` – legacy workflow normalising by downwelling
        irradiance and optional sun angle. Produces images in DN‑like units.
      * ``calibration_panels`` – scale images to reflectance using a calibrated
        reflectance panel.
      * ``auto`` – automatically select the most complete radiometric
        strategy available per image and output reflectance in the 0–1
        range. In addition to the GeoTIFF, a float32 ``.npy`` array is saved
        containing the same reflectance values for downstream analysis.

    Returns
    -------
    dict
        A report describing what was done to each file. See docstring
        for more details.
    """
    in_dir = cfg.input_folder
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Input folder not found: {in_dir}")

    tiffs = _list_tiffs(in_dir)
    if not tiffs:
        raise FileNotFoundError(f"No TIFF files found in: {in_dir}")

    method = (cfg.method or "").strip().lower()
    if method not in ("flight_metadata", "calibration_panels", "auto", "auto_plus"):
        raise ValueError("method must be 'flight_metadata', 'calibration_panels', 'auto' or 'auto_plus'")

    # Determine output folder for GeoTIFFs
    if cfg.output_folder:
        out_dir = cfg.output_folder
    else:
        if method == "flight_metadata":
            suffix = "__radiometric_flight_metadata"
        elif method == "calibration_panels":
            suffix = "__radiometric_panels"
        elif method == "auto_plus":
            suffix = "__radiometric_auto_plus"
        else:
            suffix = "__radiometric_out"
        out_dir = os.path.join(in_dir, suffix)

    _safe_mkdir(out_dir)


    report: Dict[str, object] = {
        "input_folder": in_dir,
        "output_folder": out_dir,
        "method": method,
        "physical_policy": "best_available_metadata_correction",
        "band_sensitivity_mode": str(getattr(cfg, "band_sensitivity_mode", "divide")),
        "files_total": len(tiffs),
        "files_written": 0,
        "skipped_existing": 0,
        "warnings": [],
        "per_file": [],
    }


    # Default scaling for uint16 output (reflectance-friendly defaults)
    if cfg.uint16_scale is None:
        if method == "flight_metadata":
            cfg.uint16_scale = 1.0
        elif method == "calibration_panels":
            cfg.uint16_scale = 65535.0
        else:  # auto / auto_plus
            cfg.uint16_scale = 65535.0

    # NPY outputs (auto and calibration_panels)
    npy_dir: Optional[str] = None
    if method in ("auto", "auto_plus", "calibration_panels"):
        npy_dir = os.path.join(out_dir, "__npy")
        _safe_mkdir(npy_dir)
        report["npy_folder"] = npy_dir

    # ------------------------------------------------------------------
    # Robust shielding pre-pass for auto/irradiance
    #
    # Goal: avoid reflectance "explosions" when irradiance metadata is corrupted
    # (e.g., occluded sensor -> very low irradiance -> huge gain).
    #
    # We precompute per-band robust gain stats and a rolling median over time.
    robust_index_by_path: Dict[str, Dict[str, object]] = {}
    robust_summary_by_band: Dict[str, Dict[str, object]] = {}
    if method in ("auto","auto_plus") and bool(getattr(cfg, "robust_irradiance", True)):
        irr_ref = report.get("flight_irradiance_ref_median")
        per_band_rows: Dict[str, List[Tuple[float, str, float]]] = {}  # band -> [(time_key, path, g_raw)]
        for _p in tiffs:
            _m0 = scan_tiff_metadata(_p)
            _bk = _band_key_from_meta(_m0)
            g_raw, denom, irr_raw, se, has_ref = _compute_irradiance_gain(_m0, irr_ref, cfg)
            if g_raw is None or (not np.isfinite(g_raw)) or g_raw <= 0:
                continue
            tk = _get_capture_time_key(_p)
            per_band_rows.setdefault(_bk, []).append((tk, _p, float(g_raw)))

        mad_k = float(getattr(cfg, "robust_mad_k", 6.0))
        window = int(getattr(cfg, "robust_window", 7) or 7)

        for bk, rows in per_band_rows.items():
            rows.sort(key=lambda x: x[0])
            g_arr = np.array([r[2] for r in rows], dtype=np.float64)
            if g_arr.size == 0:
                continue
            g_base = float(np.median(g_arr))
            mad = _mad(g_arr, g_base)

            if (not np.isfinite(mad)) or mad <= 0:
                lo = max(1e-12, g_base * 0.5)
                hi = g_base * 1.5
                limits_reason = "relative_fallback"
            else:
                lo = max(1e-12, g_base - mad_k * mad)
                hi = g_base + mad_k * mad
                limits_reason = "mad"

            g_smooth = _rolling_median(g_arr, window)

            n_flagged = 0
            for i, (_tk, _p, _g) in enumerate(rows):
                g0 = float(_g)
                flagged = False
                reasons: List[str] = []

                if not (lo <= g0 <= hi):
                    flagged = True
                    reasons.append("mad_limits")
                if g0 > g_base * 10.0:
                    flagged = True
                    reasons.append("too_high_vs_median")
                if g0 < g_base / 10.0:
                    flagged = True
                    reasons.append("too_low_vs_median")

                if irr_ref and irr_ref > 0:
                    if g0 < RATIO_CLAMP_MIN:
                        flagged = True
                        reasons.append("legacy_clamp_low")
                    if g0 > RATIO_CLAMP_MAX:
                        flagged = True
                        reasons.append("legacy_clamp_high")

                if flagged:
                    n_flagged += 1

                robust_index_by_path[_p] = {
                    "band_key": bk,
                    "g_raw": g0,
                    "g_smooth": float(g_smooth[i]) if np.isfinite(g_smooth[i]) else None,
                    "g_base": g_base,
                    "mad": mad,
                    "limits": [float(lo), float(hi)],
                    "flagged_outlier": bool(flagged),
                    "flag_reasons": reasons,
                    "limits_reason": limits_reason,
                    "window": window,
                }

            robust_summary_by_band[bk] = {
                "median": g_base,
                "mad": mad,
                "limits": [float(lo), float(hi)],
                "limits_reason": limits_reason,
                "window": window,
                "n_samples": int(g_arr.size),
                "n_flagged": int(n_flagged),
            }

        report["robust_irradiance_enabled"] = True
        report["robust_irradiance_by_band"] = robust_summary_by_band
    else:
        report["robust_irradiance_enabled"] = False

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Panel calibration pre-pass (calibration_panels)
    #
    # User must provide:
    #   - a folder (cfg.panel_folder) OR explicit list (cfg.panel_image_paths)
    #   - a ROI per band (cfg.panel_roi_by_band), tolerant to 'outsides'
    #   - reflectance per band (cfg.panel_reflectance_by_band) OR a global
    #     fallback (cfg.panel_reflectance)
    #
    # We compute a single robust scale per band:
    #   scale_b = rho_b / median(DN_panel_b)
    # Then reflectance = DN * scale_b, clipped to [0, 1].
    panel_scale_by_band: Dict[str, float] = {}
    panel_summary_by_band: Dict[str, Dict[str, object]] = {}
    panel_paths_set: set[str] = set()

    if method == "calibration_panels":
        # Helper to resolve panel paths from (explicit list) OR (folder)
        def _resolve_panel_paths(explicit_paths, folder_path) -> List[str]:
            paths: List[str] = []
            if explicit_paths:
                paths = [p for p in (explicit_paths or []) if p and os.path.isfile(p)]
            elif folder_path:
                pf = str(folder_path)
                if not os.path.isdir(pf):
                    raise FileNotFoundError(f"Panel folder not found: {pf}")
                paths = _list_tiffs(pf, recursive=bool(getattr(cfg, "recursive", False)))
            return paths

        def _normalize_roi_map(raw_map) -> Dict[str, Tuple[int, int, int, int]]:
            out: Dict[str, Tuple[int, int, int, int]] = {}
            if raw_map:
                for k, v in (raw_map or {}).items():
                    try:
                        bk = _norm_band_key(k)
                        x, y, w, h = v
                        out[bk] = (int(x), int(y), int(w), int(h))
                    except Exception:
                        continue
            return out

        def _normalize_refl_map(raw_map) -> Dict[str, float]:
            out: Dict[str, float] = {}
            if raw_map:
                for k, v in (raw_map or {}).items():
                    try:
                        bk = _norm_band_key(k)
                        out[bk] = float(v)
                    except Exception:
                        continue
            return out

        def _compute_scales_for_panel_set(panel_paths: List[str],
                                          roi_map_override: Optional[Dict[str, Tuple[int, int, int, int]]] = None
                                          ) -> Tuple[Dict[str, float], Dict[str, Dict[str, object]], float, set[str]]:
            """Compute a single robust scale per band for a given set of panel images.
            Returns (scale_by_band, summary_by_band, t_med, panel_paths_set).
            """
            if not panel_paths:
                return {}, {}, float("nan"), set()

            panel_paths_set_local = set(os.path.abspath(p) for p in panel_paths)

            # Normalize maps
            roi_map_local = dict(_normalize_roi_map(getattr(cfg, "panel_roi_by_band", None)))
            if roi_map_override is not None:
                roi_map_local = dict(roi_map_override)
            refl_map_local = _normalize_refl_map(getattr(cfg, "panel_reflectance_by_band", None))

            # Group by band
            per_band_panels_local: Dict[str, List[str]] = {}
            for pp in panel_paths:
                try:
                    m0 = scan_tiff_metadata(pp)
                except Exception:
                    m0 = {}
                bk = _band_key_from_meta(m0)
                per_band_panels_local.setdefault(bk, []).append(pp)

            inner_frac_local = float(getattr(cfg, "panel_roi_inner_frac", 0.80))
            dn_percentiles_local = tuple(getattr(cfg, "panel_dn_percentiles", (10.0, 90.0)))

            scales_local: Dict[str, float] = {}
            summary_local: Dict[str, Dict[str, object]] = {}

            # Median time key for the set (used for before/after interpolation)
            try:
                t_med = float(np.median(np.array([_get_capture_time_key(p) for p in panel_paths], dtype=np.float64)))
            except Exception:
                t_med = float("nan")

            for bk, plist in per_band_panels_local.items():
                roi = roi_map_local.get(bk)
                if roi is None:
                    # fallback to a centered ROI on the first panel image
                    try:
                        arr0, _, _ = _read_tiff_array_and_profile(plist[0])
                        roi = cfg.panel_roi or _default_center_roi(arr0.shape[1], arr0.shape[0], frac=0.20)
                    except Exception:
                        roi = cfg.panel_roi or (0, 0, 0, 0)

                dn_vals: List[float] = []
                for pp in plist:
                    try:
                        arr, _, _ = _read_tiff_array_and_profile(pp)
                        H, W = arr.shape[0], arr.shape[1]
                        x, y, w, h = roi
                        x = max(0, min(int(x), W - 1))
                        y = max(0, min(int(y), H - 1))
                        w = max(1, min(int(w), W - x))
                        h = max(1, min(int(h), H - y))
                        roi_clipped = (x, y, w, h)

                        # --- panel: optional black level subtraction + saturation check ---
                        bl_val = None
                        try:
                            meta_p = scan_tiff_metadata(pp)
                            mf_p = meta_p.get("meta_flat", {}) or {}
                            bl_list = mf_p.get("blacklevellist")
                            if bl_list is not None and isinstance(bl_list, str):
                                try:
                                    bl_list = ast.literal_eval(bl_list)
                                except Exception:
                                    bl_list = None
                            if isinstance(bl_list, (list, tuple)) and len(bl_list) > 0:
                                bl_val = float(np.median([float(x) for x in bl_list if x is not None]))
                            else:
                                bl_scalar = mf_p.get("blacklevel")
                                if bl_scalar is not None:
                                    bl_val = float(bl_scalar)
                        except Exception:
                            bl_val = None

                        sat_thr = float(getattr(cfg, "panel_saturation_threshold", 0.98))
                        dn_max = None
                        try:
                            ex_p = _extract_exif_params(pp)
                            bits_p = ex_p.get("bits")
                            if bits_p:
                                dn_max = float((2 ** int(bits_p)) - 1)
                        except Exception:
                            dn_max = None
                        if dn_max is None:
                            if arr.dtype == np.uint16:
                                dn_max = 65535.0
                            elif arr.dtype == np.uint8:
                                dn_max = 255.0
                            else:
                                try:
                                    dn_max = float(np.nanmax(arr))
                                except Exception:
                                    dn_max = 65535.0

                        # Saturation check on inner ROI region
                        try:
                            x0, y0, w0, h0 = roi_clipped
                            shrink_x = int(w0 * (1.0 - inner_frac_local) / 2.0)
                            shrink_y = int(h0 * (1.0 - inner_frac_local) / 2.0)
                            xi = x0 + shrink_x
                            yi = y0 + shrink_y
                            wi = max(1, w0 - 2 * shrink_x)
                            hi = max(1, h0 - 2 * shrink_y)
                            inner = arr[yi:yi+hi, xi:xi+wi]
                            if inner.size > 0 and float(np.nanmax(inner)) >= sat_thr * dn_max:
                                continue
                        except Exception:
                            pass

                        arr_for_dn = arr
                        if bl_val is not None:
                            try:
                                arr_for_dn = arr.astype(np.float32) - float(bl_val)
                                arr_for_dn = np.clip(arr_for_dn, 0.0, None)
                            except Exception:
                                arr_for_dn = arr

                        dn = _panel_dn_from_roi(arr_for_dn, roi_clipped,
                                                inner_frac=inner_frac_local,
                                                dn_percentiles=dn_percentiles_local)
                        if np.isfinite(dn) and dn > 0:
                            dn_vals.append(float(dn))
                    except Exception:
                        continue

                if not dn_vals:
                    scales_local[bk] = 1.0
                    summary_local[bk] = {
                        "status": "no_valid_panel_dn",
                        "n_images": len(plist),
                        "roi": roi,
                        "inner_frac": inner_frac_local,
                        "dn_percentiles": dn_percentiles_local,
                    }
                    continue

                dn_med = float(np.median(np.array(dn_vals, dtype=np.float64)))
                rho = float(refl_map_local.get(bk, cfg.panel_reflectance))
                if not (0.0 < rho <= 1.0):
                    rho = float(cfg.panel_reflectance)

                scale = rho / dn_med if dn_med > 0 else 1.0
                if (not np.isfinite(scale)) or scale <= 0:
                    scale = 1.0

                scales_local[bk] = float(scale)
                summary_local[bk] = {
                    "status": "ok",
                    "n_images": len(plist),
                    "n_used": len(dn_vals),
                    "rho_used": float(rho),
                    "dn_median": float(dn_med),
                    "scale": float(scale),
                    "roi": roi,
                    "inner_frac": inner_frac_local,
                    "dn_percentiles": dn_percentiles_local,
                    "example_files": [os.path.basename(p) for p in plist[:3]],
                }

            return scales_local, summary_local, t_med, panel_paths_set_local

        # Determine whether we are in BEFORE/AFTER mode (panels only)
        use_ba = bool(getattr(cfg, "panel_before_after", False))

        panel_scale_by_band = {}
        panel_summary_by_band = {}
        panel_paths_set = set()

        # store before/after info in report (and for later interpolation)
        panel_scale_by_band_before: Dict[str, float] = {}
        panel_scale_by_band_after: Dict[str, float] = {}
        panel_summary_by_band_before: Dict[str, Dict[str, object]] = {}
        panel_summary_by_band_after: Dict[str, Dict[str, object]] = {}
        t_before = float("nan")
        t_after = float("nan")

        if use_ba:
            # Resolve paths for BEFORE and AFTER. Do NOT require cfg.panel_folder.
            paths_before = _resolve_panel_paths(getattr(cfg, "panel_image_paths_before", None),
                                                getattr(cfg, "panel_folder_before", None))
            paths_after = _resolve_panel_paths(getattr(cfg, "panel_image_paths_after", None),
                                               getattr(cfg, "panel_folder_after", None))

            if not paths_before:
                raise ValueError("before/after panels enabled but no BEFORE panel images were provided.")
            if not paths_after:
                raise ValueError("before/after panels enabled but no AFTER panel images were provided.")

            roi_before = _normalize_roi_map(getattr(cfg, "panel_roi_by_band_before", None))
            roi_after = _normalize_roi_map(getattr(cfg, "panel_roi_by_band_after", None))

            panel_scale_by_band_before, panel_summary_by_band_before, t_before, set_before = _compute_scales_for_panel_set(paths_before, roi_before or None)
            panel_scale_by_band_after, panel_summary_by_band_after, t_after, set_after = _compute_scales_for_panel_set(paths_after, roi_after or None)

            # Union of panel files for skipping (when keep_panel_images is False)
            panel_paths_set = set_before.union(set_after)

            # Build a combined view (for compatibility in later per-file attachment)
            # Use BEFORE as default if AFTER missing a band.
            all_bands = set(panel_scale_by_band_before.keys()).union(panel_scale_by_band_after.keys())
            for bk in all_bands:
                s0 = panel_scale_by_band_before.get(bk, 1.0)
                s1 = panel_scale_by_band_after.get(bk, s0)
                panel_scale_by_band[bk] = float(s0)  # placeholder; actual scale chosen per-image
                # pick a summary (prefer BEFORE)
                panel_summary_by_band[bk] = panel_summary_by_band_before.get(bk) or panel_summary_by_band_after.get(bk) or {"status": "missing"}

            report["panel_before_after"] = True
            report["panel_before_time_key"] = float(t_before) if np.isfinite(t_before) else None
            report["panel_after_time_key"] = float(t_after) if np.isfinite(t_after) else None
            report["panel_summary_by_band_before"] = panel_summary_by_band_before
            report["panel_summary_by_band_after"] = panel_summary_by_band_after
            report["panel_files_total_before"] = len(paths_before)
            report["panel_files_total_after"] = len(paths_after)

        else:
            # SINGLE panel folder/list (original behaviour)
            panel_paths: List[str] = _resolve_panel_paths(getattr(cfg, "panel_image_paths", None),
                                                         getattr(cfg, "panel_folder", None))
            if not panel_paths:
                raise ValueError("calibration_panels requires panel_folder or panel_image_paths to be provided.")
            panel_scale_by_band, panel_summary_by_band, _t_med, panel_paths_set = _compute_scales_for_panel_set(panel_paths, None)
            report["panel_summary_by_band"] = panel_summary_by_band
            report["panel_files_total"] = len(panel_paths)

        # Expose combined summary for downstream per-file attachment
        if not use_ba:
            report["panel_summary_by_band"] = panel_summary_by_band


    # ------------------------------------------------------------------
    # AUTO+ global equalization (Pix4D-like, targetless)
    #
    # Idea: keep the physical per-image reflectance computation from AUTO,
    # then apply a *per-band*, *multiplicative* equalization so that each
    # image's robust median matches a global target for that band.
    # This reduces inter-image brightness variation without using panels.
    auto_plus_scale_by_path: Dict[str, float] = {}
    auto_plus_targets_by_band: Dict[str, float] = {}
    auto_plus_percentiles = tuple(getattr(cfg, "auto_plus_percentiles", (10.0, 90.0)))
    auto_plus_scale_clamp = tuple(getattr(cfg, "auto_plus_scale_clamp", (0.60, 1.6)))
    auto_plus_target_percentile = float(getattr(cfg, "auto_plus_target_percentile", 70.0))
    if method == "auto_plus":
        p_lo, p_hi = float(auto_plus_percentiles[0]), float(auto_plus_percentiles[1])
        cmin, cmax = float(auto_plus_scale_clamp[0]), float(auto_plus_scale_clamp[1])
        per_band_rows: Dict[str, List[Tuple[str, float]]] = {}  # band_key -> [(path, img_med)]
        for _p in tiffs:
            try:
                _m0 = scan_tiff_metadata(_p)
                _arr0, _profile0, _tags0 = _read_tiff_array_and_profile(_p)
                _arr0f = _arr0.astype(np.float32)
                _exif0 = _extract_exif_params(_p)
            except Exception:
                continue

            _corr0: Optional[np.ndarray] = None
            _refl_full = _compute_reflectance_full(_arr0f, _exif0, _m0)
            if _refl_full is not None:
                _corr0 = _refl_full
            else:
                _irr_ref = report.get("flight_irradiance_ref_median")
                _g_raw, _denom, _irr_raw, _se, _has_ref = _compute_irradiance_gain(_m0, _irr_ref, cfg)
                if _g_raw is not None:
                    _g_used = float(_g_raw)
                    if bool(getattr(cfg, "robust_irradiance", True)):
                        _rinfo = robust_index_by_path.get(_p)
                        if _rinfo and bool(_rinfo.get("flagged_outlier")):
                            _g_smooth = _rinfo.get("g_smooth")
                            _g_base = _rinfo.get("g_base")
                            if _g_smooth is not None and np.isfinite(float(_g_smooth)):
                                _g_used = float(_g_smooth)
                            elif _g_base is not None and np.isfinite(float(_g_base)):
                                _g_used = float(_g_base)
                    if _has_ref:
                        _g_used = float(np.clip(_g_used, RATIO_CLAMP_MIN, RATIO_CLAMP_MAX))
                    _arr_corr = _arr0f * float(_g_used)
                    _bits = _exif0.get("bits")
                    _dn_max = float(2 ** int(_bits)) if _bits else 65535.0
                    _corr0 = np.clip(_arr_corr / _dn_max, 0.0, 1.0).astype(np.float32, copy=False)
                else:
                    _corr0 = _compute_reflectance_fallback(_arr0f, _exif0)

            if _corr0 is None:
                continue

            _bk = _band_key_from_meta(_m0)
            _flat = _corr0.reshape(-1)
            _flat = _flat[np.isfinite(_flat)]
            if _flat.size < 10:
                continue
            try:
                _lo = float(np.percentile(_flat, p_lo))
                _hi = float(np.percentile(_flat, p_hi))
                _mid = _flat[(_flat >= _lo) & (_flat <= _hi)]
                if _mid.size == 0:
                    _mid = _flat
                _med = float(np.median(_mid))
            except Exception:
                continue
            if np.isfinite(_med) and _med > 0:
                per_band_rows.setdefault(_bk, []).append((_p, _med))

        # Compute global targets and per-image scales
        for _bk, _rows in per_band_rows.items():
            _meds = np.array([r[1] for r in _rows], dtype=np.float64)
            if _meds.size == 0:
                continue
            _q = float(auto_plus_target_percentile)
            if (not np.isfinite(_q)):
                _q = 50.0
            _q = float(np.clip(_q, 0.0, 100.0))
            _tgt = float(np.percentile(_meds, _q))
            if (not np.isfinite(_tgt)) or _tgt <= 0:
                continue
            auto_plus_targets_by_band[_bk] = float(_tgt)
            for _p, _med in _rows:
                _s = float(_tgt / max(1e-12, float(_med)))
                if (not np.isfinite(_s)) or _s <= 0:
                    _s = 1.0
                _s = float(np.clip(_s, cmin, cmax))
                auto_plus_scale_by_path[_p] = _s

        report["auto_plus_targets_by_band"] = auto_plus_targets_by_band
        report["auto_plus_percentiles"] = [p_lo, p_hi]
        report["auto_plus_scale_clamp"] = [cmin, cmax]
        report["auto_plus_target_percentile"] = float(auto_plus_target_percentile)
        report["auto_plus_scales_computed"] = len(auto_plus_scale_by_path)

    # Process all TIFFs
    for path in tiffs:
        name = os.path.basename(path)
        out_path = os.path.join(out_dir, name)
        npy_path = os.path.join(npy_dir, os.path.splitext(name)[0] + ".npy") if (method in ("auto","auto_plus","calibration_panels") and npy_dir) else None

        if (not cfg.overwrite) and os.path.exists(out_path):
            report["skipped_existing"] += 1
            continue

        # Read metadata and pixel data
        meta = scan_tiff_metadata(path)
        arr, profile, _tags_src = _read_tiff_array_and_profile(path)
        arr_f = arr.astype(np.float32)
        exif_params = _extract_exif_params(path)

        applied: Dict[str, object] = {"method": method}

        # --- best available metadata correction (AUTO/AUTO+) ---
        _sens = _pick_band_sensitivity(meta)
        physical_available = {
            "blacklevel": exif_params.get("black_level") is not None,
            "bandsensitivity": _sens is not None,
            "vignetting": False,
            "irradiance": _pick_irradiance(meta) is not None,
        }
        physical_applied = {
            "blacklevel": False,
            "bandsensitivity": None,  # "divide"|"multiply"|None
            "irradiance": False,
            "global_equalization": False,
        }

        corrected: Optional[np.ndarray] = None

        if method == "flight_metadata":
            decision_path: List[str] = ["flight_metadata"]
            irr = _pick_irradiance(meta)
            if irr is None or irr <= 0:
                corrected = arr_f
                applied["status"] = "pass_through_no_irradiance"
                decision_path.append("no_irradiance->pass_through")
            else:
                denom = float(irr)
                applied["irradiance_raw"] = float(irr)
                physical_applied["irradiance"] = True
                decision_path.append("irradiance_found")
                if cfg.apply_sun_angle:
                    elev = _pick_solar_elevation_deg(meta)
                    if elev is not None:
                        denom *= max(1e-6, math.sin(math.radians(float(elev))))
                        applied["solar_elevation_deg"] = float(elev)
                        decision_path.append("sun_angle_applied")
                    else:
                        decision_path.append("sun_angle_missing")
                irr_ref = report.get("flight_irradiance_ref_median")
                if irr_ref is None or irr_ref <= 0:
                    corrected = arr_f / max(1e-12, denom)
                    applied["status"] = "normalized_divide_by_irradiance_no_ref"
                    decision_path.append("no_ref->divide")
                else:
                    ratio = float(irr_ref) / max(1e-12, denom)
                    applied["irradiance_ref_median"] = float(irr_ref)
                    applied["ratio_raw"] = float(ratio)
                    ratio_clamped = ratio
                    if ratio < RATIO_CLAMP_MIN:
                        ratio_clamped = RATIO_CLAMP_MIN
                        applied["ratio_clamped"] = float(ratio_clamped)
                        applied["ratio_clamp_reason"] = "low"
                        decision_path.append("ratio_clamped_low")
                    elif ratio > RATIO_CLAMP_MAX:
                        ratio_clamped = RATIO_CLAMP_MAX
                        applied["ratio_clamped"] = float(ratio_clamped)
                        applied["ratio_clamp_reason"] = "high"
                        decision_path.append("ratio_clamped_high")
                    else:
                        decision_path.append("ratio_in_range")
                    corrected = arr_f * float(ratio_clamped)
                    applied["status"] = "normalized_by_ratio_ref_over_img"
                    decision_path.append("ref_ratio_applied")
            applied["decision_path"] = decision_path

        elif method == "calibration_panels":
            band_key = _band_key_from_meta(meta)
            # Panel scale selection:
            # - single: use precomputed per-band scale
            # - before/after: interpolate per image capture time between before and after scales (clamped)
            if bool(getattr(cfg, "panel_before_after", False)):
                tk_img = _get_capture_time_key(path)
                s0 = panel_scale_by_band_before.get(band_key, panel_scale_by_band_before.get("band_unknown", 1.0))
                s1 = panel_scale_by_band_after.get(band_key, panel_scale_by_band_after.get("band_unknown", s0))
                # Clamp or interpolate
                if (not np.isfinite(t_before)) or (not np.isfinite(t_after)) or (t_after <= t_before):
                    scale = float(s0)
                    applied["panel_interp"] = {"mode": "before_after_invalid_time_range", "scale_used": float(scale)}
                else:
                    if tk_img <= t_before:
                        scale = float(s0)
                        applied["panel_interp"] = {"mode": "clamp_before", "scale_used": float(scale)}
                    elif tk_img >= t_after:
                        scale = float(s1)
                        applied["panel_interp"] = {"mode": "clamp_after", "scale_used": float(scale)}
                    else:
                        w = float((tk_img - t_before) / (t_after - t_before))
                        w = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)
                        scale = float((1.0 - w) * float(s0) + w * float(s1))
                        applied["panel_interp"] = {"mode": "linear", "w": float(w), "scale_before": float(s0), "scale_after": float(s1), "scale_used": float(scale)}
            else:
                scale = panel_scale_by_band.get(band_key)
                if scale is None:
                    scale = panel_scale_by_band.get("band_unknown", 1.0)
                        # Apply panel scale to (DN - black_level) when available (best-effort)
            bl_val = None
            try:
                mf = (meta.get("meta_flat", {}) or {})
                bl_list = mf.get("blacklevellist")
                if bl_list is not None and isinstance(bl_list, str):
                    try:
                        bl_list = ast.literal_eval(bl_list)
                    except Exception:
                        bl_list = None
                if isinstance(bl_list, (list, tuple)) and len(bl_list) > 0:
                    bl_val = float(np.median([float(x) for x in bl_list if x is not None]))
                else:
                    bl_scalar = mf.get("blacklevel")
                    if bl_scalar is not None:
                        bl_val = float(bl_scalar)
            except Exception:
                bl_val = None

            arr_panel = arr_f
            if bl_val is not None:
                try:
                    arr_panel = np.clip(arr_f - float(bl_val), 0.0, None)
                    applied["black_level"] = float(bl_val)
                    applied["black_level_applied"] = True
                except Exception:
                    arr_panel = arr_f
                    applied["black_level_applied"] = False
            else:
                applied["black_level_applied"] = False

            corrected = np.clip(arr_panel * float(scale), 0.0, 1.0).astype(np.float32, copy=False)
            applied["panel_scale"] = float(scale)
            applied["band_key"] = band_key
            # Attach panel stats if available
            ps = panel_summary_by_band.get(band_key) if isinstance(panel_summary_by_band, dict) else None
            if ps:
                applied["panel_rho_used"] = ps.get("rho_used")
                applied["panel_dn_median"] = ps.get("dn_median")
                applied["panel_roi"] = ps.get("roi")
                applied["panel_inner_frac"] = ps.get("inner_frac")
                applied["panel_dn_percentiles"] = ps.get("dn_percentiles")
            applied["status"] = "scaled_by_panel"

            if (not cfg.keep_panel_images) and (os.path.abspath(path) in panel_paths_set):
                report["per_file"].append({
                    "file": name,
                    "output": None,
                    "metadata": {
                        "band_name": meta.get("band_name"),
                        "central_wavelength_nm": meta.get("central_wavelength_nm"),
                        "rig_camera_index": meta.get("rig_camera_index"),
                    },
                    "applied": {**applied, "status": "skipped_panel_image"},
                })
                continue

        else:  # auto
            decision_path: List[str] = []

            refl_full = _compute_reflectance_full(arr_f, exif_params, meta)
            if refl_full is not None:
                corrected = refl_full
                applied["strategy"] = "full"
                decision_path.append("full")
                # Full model used physical metadata (requires black level + irradiance)
                physical_applied["blacklevel"] = bool(physical_available.get("blacklevel"))
                physical_applied["irradiance"] = True
            else:
                irr_ref = report.get("flight_irradiance_ref_median")
                g_raw, denom, irr_raw, se, has_ref = _compute_irradiance_gain(meta, irr_ref, cfg)

                if g_raw is not None:
                    g_used = float(g_raw)
                    flagged_outlier = False
                    replacement_reason = None

                    if bool(getattr(cfg, "robust_irradiance", True)):
                        rinfo = robust_index_by_path.get(path)
                        if rinfo and bool(rinfo.get("flagged_outlier")):
                            flagged_outlier = True
                            g_smooth = rinfo.get("g_smooth")
                            g_base = rinfo.get("g_base")
                            if g_smooth is not None and np.isfinite(float(g_smooth)):
                                g_used = float(g_smooth)
                                replacement_reason = "rolling_median"
                            elif g_base is not None and np.isfinite(float(g_base)):
                                g_used = float(g_base)
                                replacement_reason = "band_median"
                            else:
                                replacement_reason = "no_replacement_available"

                    if has_ref:
                        g_used = float(np.clip(g_used, RATIO_CLAMP_MIN, RATIO_CLAMP_MAX))

                    bl = exif_params.get("black_level")
                    if bl is not None:
                        try:
                            arr_corr = (arr_f - float(bl)) * float(g_used)
                            physical_applied["blacklevel"] = True
                            applied["black_level"] = float(bl)
                        except Exception:
                            arr_corr = arr_f * float(g_used)
                    else:
                        arr_corr = arr_f * float(g_used)
                    physical_applied["irradiance"] = True
                    bits = exif_params.get("bits")
                    dn_max = float(2 ** int(bits)) if bits else 65535.0
                    corrected = np.clip(arr_corr / dn_max, 0.0, 1.0).astype(np.float32, copy=False)

                    applied["strategy"] = "irradiance"
                    decision_path.append("irradiance")
                    if irr_ref:
                        applied["irradiance_ref_median"] = float(irr_ref)
                    applied["irradiance_raw"] = float(irr_raw) if irr_raw is not None else None
                    if se is not None:
                        applied["solar_elevation_deg"] = float(se)

                    applied["g_raw"] = float(g_raw)
                    applied["g_used"] = float(g_used)
                    applied["flagged_outlier"] = bool(flagged_outlier)
                    applied["replacement_reason"] = replacement_reason

                    if bool(getattr(cfg, "robust_irradiance", True)):
                        rinfo = robust_index_by_path.get(path)
                        if rinfo:
                            applied["band_key"] = rinfo.get("band_key")
                            applied["robust_limits"] = rinfo.get("limits")
                            applied["robust_mad"] = rinfo.get("mad")
                            applied["robust_flag_reasons"] = rinfo.get("flag_reasons")
                else:
                    corrected = _compute_reflectance_fallback(arr_f, exif_params)
                    applied["strategy"] = "fallback"
                    decision_path.append("fallback")
                    if exif_params.get("black_level") is not None:
                        physical_applied["blacklevel"] = True
                        try:
                            applied["black_level"] = float(exif_params.get("black_level"))
                        except Exception:
                            pass

            applied["decision_path"] = decision_path

        if corrected is None:
            report["warnings"].append(f"Failed to correct file {name}; skipping.")
            continue

        
        # Apply optional Band Sensitivity correction (AUTO/AUTO+ only)
        if method in ("auto", "auto_plus"):
            bs_mode = str(getattr(cfg, "band_sensitivity_mode", "divide") or "divide").strip().lower()
            if bs_mode in ("off", "none", "false", "0"):
                bs_mode = "off"
            if (_sens is not None) and np.isfinite(float(_sens)) and float(_sens) > 0 and bs_mode in ("divide", "multiply"):
                try:
                    if bs_mode == "divide":
                        corrected = corrected / float(_sens)
                    else:
                        corrected = corrected * float(_sens)
                    corrected = np.clip(corrected, 0.0, 1.0).astype(np.float32, copy=False)
                    physical_applied["bandsensitivity"] = bs_mode
                    applied["band_sensitivity"] = float(_sens)
                    applied["band_sensitivity_mode"] = bs_mode
                except Exception as _e:
                    applied["band_sensitivity_error"] = repr(_e)
            else:
                # Explicitly record why it wasn't applied (transparency)
                if bs_mode == "off":
                    applied["band_sensitivity_skipped_reason"] = "disabled_by_config"
                elif _sens is None:
                    applied["band_sensitivity_skipped_reason"] = "metadata_missing"
                else:
                    applied["band_sensitivity_skipped_reason"] = "invalid_value_or_mode"

        applied["physical_available"] = physical_available
        applied["physical_applied"] = physical_applied

# Apply AUTO+ equalization after computing reflectance
        if method == "auto_plus":
            physical_applied["global_equalization"] = True
            try:
                _bk = _band_key_from_meta(meta)
            except Exception:
                _bk = "band_unknown"
            _scale = auto_plus_scale_by_path.get(path, 1.0)
            if (not np.isfinite(float(_scale))) or float(_scale) <= 0:
                _scale = 1.0
            # record robust median stats (before scaling)
            try:
                _p_lo, _p_hi = float(auto_plus_percentiles[0]), float(auto_plus_percentiles[1])
                _flat = corrected.reshape(-1)
                _flat = _flat[np.isfinite(_flat)]
                _img_med = None
                if _flat.size >= 10:
                    _lo = float(np.percentile(_flat, _p_lo))
                    _hi = float(np.percentile(_flat, _p_hi))
                    _mid = _flat[(_flat >= _lo) & (_flat <= _hi)]
                    if _mid.size == 0:
                        _mid = _flat
                    _img_med = float(np.median(_mid))
                applied["auto_plus_img_median"] = _img_med
                applied["auto_plus_target_median"] = auto_plus_targets_by_band.get(_bk)
                applied["auto_plus_scale"] = float(_scale)
                applied["auto_plus_band_key"] = _bk
            except Exception:
                applied["auto_plus_scale"] = float(_scale)
                applied["auto_plus_band_key"] = _bk
            corrected = np.clip(corrected * float(_scale), 0.0, 1.0).astype(np.float32, copy=False)

        corrected_out = _to_output_dtype(corrected, cfg)

        try:
            _write_tiff(out_path, corrected_out, profile, tags={}, src_path=path)
        except Exception as e:
            report["warnings"].append(f"Failed to write {name}: {e!r}")
            continue

        if method in ("auto","auto_plus","calibration_panels") and npy_path:
            try:
                np.save(npy_path, corrected.astype(np.float32, copy=False))
                applied["npy"] = os.path.basename(npy_path)
            except Exception as e:
                report["warnings"].append(f"Failed to save NPY for {name}: {e!r}")

        report["files_written"] += 1

        report["per_file"].append({
            "file": name,
            "output": out_path,
            "metadata": {
                "band_name": meta.get("band_name"),
                "central_wavelength_nm": meta.get("central_wavelength_nm"),
                "rig_camera_index": meta.get("rig_camera_index"),
            },
            "applied": applied,
        })

    # Write report JSON to output folder
    try:
        report_path = os.path.join(out_dir, "radiometric_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        report["report_json"] = report_path
    except Exception as e:
        report["warnings"].append(f"Failed to write report JSON: {e!r}")

    return report