# geo_import.py — GeoImport with .DAT/.EXIF → ODM (Docker) → multiband orthomosaic
# Workflow: single selection of the folder containing the flight images;
# outputs are written to "<folder>/__odm_outputs".
# Profiles: FAST and FULL.
#
# Key points:
#  - Robust alignment between orthomosaics when georeferencing is missing or inconsistent
#    (phase correlation + ECC affine in pyramid, with safe fallback).
#  - Integration with calibrar_cubo_por_paineis_com_bandas_gui
#    (visible rectangular ROI selection + ENTER/ESC).

import os
import re
import json
import subprocess
from collections import defaultdict

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, Toplevel, Label, Button, StringVar
from tkinter import ttk

import cv2
import rasterio
from rasterio.warp import reproject, Resampling

# tifffile gives access to EXIF/GPS/XMP blocks that rasterio often doesn't expose
import tifffile
import xml.etree.ElementTree as ET

# =============== Rasterio warnings for non-georeferenced images ===============
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# =============================================================================
# Auto band metadata inference (module-level)
#
# This block enables vendor-agnostic wavelength detection without user input.
# It is intentionally defined at module scope so geoimport_wizard_gui can always
# call infer_nm_map_from_groups(...).
# =============================================================================

_WL_KEYS = (
    "WAVELENGTH",
    "wavelength",
    "CENTER_WAVELENGTH",
    "center_wavelength",
    "CENTRAL_WAVELENGTH",
    "central_wavelength",
    "CentralWavelength",
    "Xmp.Camera.CentralWavelength",
    "BandName",
    "BANDNAME",
    "BAND_NAME",
    "band_name",
    "Xmp.Camera.BandName",
    "DESCRIPTION",
    "description",
)

_SPECTRAL_NAMES = [
    (450, "Blue"),
    (475, "Blue"),
    (550, "Green"),
    (560, "Green"),
    (650, "Red"),
    (660, "Red"),
    (668, "Red"),
    (715, "RedEdge"),
    (717, "RedEdge"),
    (730, "RedEdge"),
    (735, "RedEdge"),
    (790, "NIR"),
    (840, "NIR"),
    (842, "NIR"),
    (860, "NIR"),
    (1100, "SWIR"),
]


def _parse_first_float(text):
    if text is None:
        return None
    s = str(text)
    m = re.search(r"(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _infer_name_from_nm(nm, tol=25.0):
    if nm is None:
        return None
    best = None
    best_d = 1e18
    for ref_nm, name in _SPECTRAL_NAMES:
        d = abs(float(nm) - float(ref_nm))
        if d < best_d:
            best = name
            best_d = d
    return best if best_d <= float(tol) else None


def _read_sidecar_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return None


def _extract_nm_from_sidecar(tif_path):
    """Try sidecar files (.dat/.xmp/.xml/.json/.txt) to infer wavelength (nm)."""
    base, _ = os.path.splitext(tif_path)
    candidates = [
        base + ".dat",
        base + ".xmp",
        base + ".xml",
        base + ".json",
        base + ".txt",
    ]

    for p in candidates:
        if not os.path.exists(p):
            continue

        # .dat via existing parser (if present)
        if p.endswith(".dat") and "_parse_dat_file" in globals():
            try:
                d = _parse_dat_file(p)
                nm = d.get("band_nm")
                if nm:
                    return float(nm)
            except Exception:
                pass

        # JSON recursive search
        if p.endswith(".json"):
            try:
                obj = json.loads(_read_sidecar_text(p) or "")
                stack = [obj]
                while stack:
                    cur = stack.pop()
                    if isinstance(cur, dict):
                        for k, v in cur.items():
                            if str(k) in _WL_KEYS:
                                nm = _parse_first_float(v)
                                if nm:
                                    return nm
                            stack.append(v)
                    elif isinstance(cur, list):
                        stack.extend(cur)
            except Exception:
                pass

        txt = _read_sidecar_text(p)
        if not txt:
            continue

        # quick text scan around known keys
        for key in _WL_KEYS:
            if key in txt:
                idx = txt.find(key)
                snippet = txt[max(0, idx - 80) : idx + 220]
                nm = _parse_first_float(snippet)
                if nm and 300 <= nm <= 2500:
                    return nm

        # generic number scan (last resort)
        nm = _parse_first_float(txt)
        if nm and 300 <= nm <= 2500:
            return nm

    return None


def _extract_nm_from_tif_metadata(tif_path):
    """Try to extract wavelength from GeoTIFF tags/descriptions using rasterio."""
    try:
        with rasterio.open(tif_path) as ds:
            if ds.descriptions and ds.descriptions[0]:
                nm = _parse_first_float(ds.descriptions[0])
                if nm and 300 <= nm <= 2500:
                    return nm

            for tags in (ds.tags(1) or {}, ds.tags() or {}):
                for k in _WL_KEYS:
                    if k in tags:
                        nm = _parse_first_float(tags.get(k))
                        if nm and 300 <= nm <= 2500:
                            return nm

            tags_all = ds.tags() or {}
            for v in tags_all.values():
                nm = _parse_first_float(v)
                if nm and 300 <= nm <= 2500:
                    return nm
    except Exception:
        return None
    return None




def infer_nm_map_from_groups(groups, meta_cache=None, sample_per_band=6):
    """
    Infer band_key -> wavelength (nm) and labels without user input.

    Priority per image:
      1) cached metadata (meta_cache[path]['central_wavelength_nm'|'band_name'|'rig_camera_index'])
      2) sidecar/.tif metadata miners (existing helpers)
      3) band_key looks like nm (e.g., '475')

    Returns:
      nm_map: dict band_key -> float|None
      label_map: dict band_key -> str
    """
    nm_map = {}
    label_map = {}

    def _meta_for_path(p):
        if meta_cache and p in meta_cache:
            return meta_cache.get(p) or {}
        return {}

    for band_key, paths in groups.items():
        nms = []
        names = []
        rigs = []
        for p in (paths or [])[: int(sample_per_band)]:
            m = _meta_for_path(p)
            nm = m.get('central_wavelength_nm')
            if nm is None:
                # fall back to existing miners if present
                if '_extract_nm_from_sidecar' in globals():
                    nm = _extract_nm_from_sidecar(p)
                if nm is None and '_extract_nm_from_tif_metadata' in globals():
                    nm = _extract_nm_from_tif_metadata(p)
            if nm is not None:
                try:
                    nms.append(float(nm))
                except Exception:
                    pass

            bn = m.get('band_name')
            if bn:
                names.append(str(bn))

            ri = m.get('rig_camera_index')
            if ri is not None:
                try:
                    rigs.append(int(ri))
                except Exception:
                    pass

        nm = None
        if nms:
            nms_sorted = sorted(nms)
            nm = nms_sorted[len(nms_sorted) // 2]  # median

        # fallback: band_key itself might be nm
        if nm is None:
            s = str(band_key).strip()
            if s.isdigit():
                v = float(s)
                if v >= 100:
                    nm = v

        nm_map[band_key] = nm

        # label
        # prefer explicit band_name from metadata if stable
        label = None
        if names:
            # choose most common name
            from collections import Counter
            label = Counter(names).most_common(1)[0][0]

        if label is None:
            if ' _infer_name_from_nm' in globals():
                try:
                    label = _infer_name_from_nm(nm)
                except Exception:
                    label = None

        if nm is None:
            label_map[band_key] = f"Band {band_key}"
        else:
            nm_i = int(round(float(nm)))
            if label:
                # If label already includes nm, don't duplicate
                if re.search(r"\b\d{3,4}\b", str(label)):
                    label_map[band_key] = str(label)
                else:
                    label_map[band_key] = f"{label} {nm_i}nm"
            else:
                label_map[band_key] = f"{nm_i}nm"

    return nm_map, label_map


def _info(msg, title="Info"):
    messagebox.showinfo(title, msg)


def _warn(msg, title="Warning"):
    messagebox.showwarning(title, msg)


def _error(msg, title="Error"):
    messagebox.showerror(title, msg)


# -------------------------
# Docker / ODM check
# -------------------------
def _docker_available():
    """Return True if Docker is available in the system PATH."""
    try:
        subprocess.run(
            ["docker", "--version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


# -------------------------
# Group files by band index
# -------------------------
_TIF_RE = re.compile(r".+\.(tif|tiff)$", re.IGNORECASE)
# Patterns to extract band index: ..._1.tif  / ..._475.tif  / ..._band3.tif
_IDX_PATTERNS = [
    re.compile(r".*_(\d+)\.(?:tif|tiff)$", re.IGNORECASE),
    re.compile(r".*band[_-]?(\d+)\.(?:tif|tiff)$", re.IGNORECASE),
]


def scan_flight_folder(folder):
    """
    Scan the selected folder and group TIFF files by band index.

    Returns
    -------
    dict
        Mapping band_key -> list of image paths.
    """
    groups = defaultdict(list)
    for root, _, files in os.walk(folder):
        for fn in files:
            if not _TIF_RE.match(fn):
                continue
            idx = None
            for pat in _IDX_PATTERNS:
                m = pat.match(fn)
                if m:
                    idx = m.group(1)
                    break
            if idx is None:
                idx = "unknown"
            groups[idx].append(os.path.join(root, fn))
    for k in groups:
        groups[k] = sorted(groups[k])
    return dict(groups)


# -------------------------
# TIFF metadata scanner (vendor-agnostic)
# -------------------------
def scan_tiff_metadata(tif_path):
    """Scan a TIFF/GeoTIFF and return metadata useful for band labeling and
    radiometric/panel corrections.

    Notes
    -----
    - Many multispectral/hyperspectral cameras store band info in XMP or EXIF.
      GeoTIFF writers may expose some of that via Rasterio's tags/description.
    - For ENVI-style hyperspectral exports, wavelength lists are commonly stored
      in a sidecar `.hdr` file.

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
        - dls (dict)  # downwelling light sensor / panel-related fields
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
    }


    # ---- sanity checks ----
    try:
        meta["raw_tags"]["_path"] = os.path.abspath(tif_path)
        meta["raw_tags"]["_exists"] = str(os.path.exists(tif_path))
        if os.path.exists(tif_path):
            meta["raw_tags"]["_size_bytes"] = str(os.path.getsize(tif_path))
    except Exception as e:
        meta["raw_tags"]["_sanity_error"] = repr(e)
    # ---- helpers ----
    def _norm(s):
        return str(s).strip() if s is not None else ""

    def _try_float(x):
        try:
            return float(x)
        except Exception:
            return None

    def _search_float_in_text(text, keys):
        if not text:
            return None
        t = str(text)
        for k in keys:
            if k in t:
                idx = t.find(k)
                snippet = t[max(0, idx - 80) : idx + 240]
                v = _parse_first_float(snippet)
                if v is not None:
                    return v
        # fallback: first float
        v = _parse_first_float(t)
        return v

    def _ns_to_prefix(ns: str) -> str:
        """Map known XMP namespaces to stable prefixes (Pix4D-style)."""
        ns = (ns or "").lower()
        if "pix4d.com/camera" in ns:
            return "Camera"
        if "pix4d.com/dls" in ns:
            return "DLS"
        if "micasense" in ns:
            return "MicaSense"
        if "dji" in ns:
            return "DJI"
        if "parrot" in ns:
            return "Parrot"
        return "XMP"

    def _parse_xmp_packet(xmp_text: str) -> dict:
        """Parse an XMP packet into a flat dict key->text.

        We keep keys as '<Prefix>:<LocalName>' so different vendors can still
        be mined with heuristics (e.g., 'Camera:CentralWavelength',
        'DLS:Irradiance', etc.).
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

    # ---- 1) ENVI header sidecar (.hdr) for hyperspectral ----
    try:
        base, _ = os.path.splitext(tif_path)
        hdr_path = base + ".hdr"
        if os.path.exists(hdr_path):
            txt = _read_sidecar_text(hdr_path) or ""
            # wavelength units may be micrometers or nanometers
            units = None
            m_units = re.search(r"wavelength\s+units\s*=\s*\{([^}]*)\}", txt, re.I)
            if m_units:
                units = m_units.group(1).strip().lower()
            m = re.search(r"wavelength\s*=\s*\{([^}]*)\}", txt, re.I | re.S)
            if m:
                nums = re.findall(r"-?\d+(?:\.\d+)?", m.group(1))
                w = [float(n) for n in nums]
                # heuristics: ENVI often uses micrometers (0.485, 0.560...)
                if units and "micro" in units:
                    w = [v * 1000.0 for v in w]
                else:
                    # if values look like micrometers, convert to nm
                    if w and max(w) < 20:
                        w = [v * 1000.0 for v in w]
                meta["wavelength_list_nm"] = w
    except Exception as e:
        meta["raw_tags"]["_hdr_error"] = repr(e)
        pass

    # ---- 2) Rasterio tags/descriptions ----
    if rasterio is not None:
        try:
            with rasterio.open(tif_path) as ds:
                # Collect tags (dataset + band 1)
                tags = {}
                try:
                    tags.update(ds.tags() or {})
                except Exception:
                    pass
                try:
                    tags.update(ds.tags(1) or {})
                except Exception:
                    pass

                # Preserve raw tags for debugging
                meta["raw_tags"].update({str(k): str(v) for k, v in tags.items()})

                # camera model hints
                for k in ("Model", "model", "CameraModel", "camera_model", "Make", "make"):
                    if k in tags and tags.get(k):
                        meta["camera_model"] = _norm(tags.get(k))
                        break

                # band description/name
                if ds.descriptions and ds.descriptions[0]:
                    meta["band_name"] = _norm(ds.descriptions[0])

                # wavelength from common tag keys
                wl_keys = (
                    "Xmp.Camera.CentralWavelength",
                    "CentralWavelength",
                    "CENTER_WAVELENGTH",
                    "center_wavelength",
                    "CENTRAL_WAVELENGTH",
                    "central_wavelength",
                    "WAVELENGTH",
                    "wavelength",
                )
                for k in wl_keys:
                    if k in tags and tags.get(k):
                        v = _parse_first_float(tags.get(k))
                        if v is not None:
                            # some XMP store micrometers
                            if v < 50:
                                v = v * 1000.0
                            meta["central_wavelength_nm"] = v
                            break
                if meta["central_wavelength_nm"] is None:
                    # sometimes wavelength is embedded in description
                    v = _parse_first_float(meta.get("band_name"))
                    if v is not None:
                        if v < 50:
                            v = v * 1000.0
                        meta["central_wavelength_nm"] = v

                # BandName tag
                for k in ("Xmp.Camera.BandName", "BandName", "band_name", "BAND_NAME"):
                    if k in tags and tags.get(k):
                        meta["band_name"] = meta["band_name"] or _norm(tags.get(k))
                        break

                # FWHM/Bandwidth
                for k in ("FWHM", "fwhm", "WavelengthFWHM", "Xmp.Camera.WavelengthFWHM"):
                    if k in tags and tags.get(k):
                        bw = _parse_first_float(tags.get(k))
                        if bw is not None:
                            if bw < 50:
                                bw = bw * 1000.0
                            meta["fwhm_nm"] = bw
                            break

                # RigCameraIndex (DJI/Parrot-like)
                for k in ("RigCameraIndex", "rigcamerindex", "Xmp.Camera.RigCameraIndex"):
                    if k in tags and tags.get(k):
                        idx = _parse_first_float(tags.get(k))
                        if idx is not None:
                            meta["rig_camera_index"] = int(round(idx))
                            break

                # DLS / panel correction hints (keep raw numbers)
                # Examples seen in the wild: Xmp.DLS.*, DLS:*, Irradiance, etc.
                for k, v in tags.items():
                    kl = str(k).lower()
                    if "dls" in kl or "irradi" in kl or "downwelling" in kl:
                        fv = _parse_first_float(v)
                        if fv is not None:
                            meta["dls"][str(k)] = fv

                # If this is a multiband TIFF and we have an ENVI wavelength list,
                # keep it for later use.
                if ds.count and ds.count > 1 and meta.get("wavelength_list_nm"):
                    if len(meta["wavelength_list_nm"]) != ds.count:
                        # mismatch - keep but mark (caller can decide)
                        meta["raw_tags"]["_wavelength_list_mismatch"] = f"hdr={len(meta['wavelength_list_nm'])}, tiff={ds.count}"

        except Exception as e:
            meta["raw_tags"]["_rasterio_error"] = repr(e)
            pass

    # ---- 2b) TIFF EXIF/GPS/XMP blocks via tifffile (vendor-agnostic) ----
    # Rasterio/GDAL often do not expose XMP/EXIF fields for non-GeoTIFF imagery.
    # tifffile lets us mine those reliably.
    try:
        with tifffile.TiffFile(tif_path) as tif:
            page = tif.pages[0]
            ttags = page.tags

            # Make/Model
            for k in ("Make", "Model"):
                if k in ttags and ttags[k].value:
                    meta["camera_model"] = meta["camera_model"] or _norm(ttags[k].value)

            # EXIF dict
            if "ExifTag" in ttags and isinstance(ttags["ExifTag"].value, dict):
                exif = ttags["ExifTag"].value
                for ek, ev in exif.items():
                    if ev is None:
                        continue
                    key = f"EXIF:{ek}"
                    if key not in meta["raw_tags"]:
                        meta["raw_tags"][key] = str(ev)

            # GPS dict
            if "GPSTag" in ttags and isinstance(ttags["GPSTag"].value, dict):
                gps = ttags["GPSTag"].value
                for gk, gv in gps.items():
                    meta["raw_tags"][f"GPS:{gk}"] = str(gv)

            # XMP packet
            xmp_text = None
            if "XMP" in ttags and ttags["XMP"].value:
                xv = ttags["XMP"].value
                if isinstance(xv, (bytes, bytearray)):
                    xmp_text = xv.decode("utf-8", "ignore")
                else:
                    xmp_text = str(xv)

            if xmp_text:
                xmp = _parse_xmp_packet(xmp_text)
                for k, v in xmp.items():
                    meta["raw_tags"][f"XMP:{k}"] = v

                # Prefer a more specific rig/camera model when available
                rig = xmp.get("Camera:RigName") or xmp.get("Xmp.Camera.RigName")
                if rig:
                    # If we only got a generic make/model, overwrite it with the rig name
                    meta["camera_model"] = _norm(rig)

                # Pix4D-style Camera tags
                if not meta["band_name"]:
                    bn = xmp.get("Camera:BandName") or xmp.get("Xmp.Camera.BandName")
                    if bn:
                        meta["band_name"] = _norm(bn)

                if meta["central_wavelength_nm"] is None:
                    cw = xmp.get("Camera:CentralWavelength") or xmp.get("Xmp.Camera.CentralWavelength")
                    v = _parse_first_float(cw)
                    if v is not None:
                        meta["central_wavelength_nm"] = v

                if meta["fwhm_nm"] is None:
                    bw = xmp.get("Camera:WavelengthFWHM") or xmp.get("Xmp.Camera.WavelengthFWHM")
                    v = _parse_first_float(bw)
                    if v is not None:
                        meta["fwhm_nm"] = v

                if meta["rig_camera_index"] is None:
                    rc = xmp.get("Camera:RigCameraIndex") or xmp.get("Xmp.Camera.RigCameraIndex") or xmp.get("RigCameraIndex")
                    v = _parse_first_float(rc)
                    if v is not None:
                        meta["rig_camera_index"] = int(round(v))

                # DLS / irradiance fields
                for k, v in xmp.items():
                    kl = str(k).lower()
                    if "dls" in kl or "irradi" in kl or "downwelling" in kl:
                        fv = _parse_first_float(v)
                        if fv is not None:
                            meta["dls"][k] = fv

    except Exception as e:
        meta["raw_tags"]["_tifffile_error"] = repr(e)
        pass

    # ---- 3) Sidecar metadata (.dat/.xmp/.xml/.json/.txt) ----
    try:
        if meta["central_wavelength_nm"] is None:
            nm = _extract_nm_from_sidecar(tif_path)
            if nm is not None:
                meta["central_wavelength_nm"] = float(nm)
    except Exception:
        pass

    return meta




def varrer_metadados_tiff(image_paths):
    """Scan a list of TIFFs and group them into bands.

    Returns:
      groups: dict[band_key] -> list[path]
      meta_cache: dict[path] -> meta dict

    Band key strategy (Pix4D-like):
      1) central_wavelength_nm (best)
      2) normalized band_name (or nm embedded in name)
      3) rig_camera_index
      4) filename fallback

    This makes ordering and labeling stable across cameras.
    """
    groups = {}
    meta_cache = {}

    for p in image_paths:
        try:
            meta = scan_tiff_metadata(p)
        except Exception:
            meta = {"raw_tags": {"_scan_error": True}}

        meta_cache[p] = meta
        nm = meta.get('central_wavelength_nm')
        bn = meta.get('band_name')
        ri = meta.get('rig_camera_index')

        band_key = None

        if nm is not None:
            try:
                band_key = float(nm)
            except Exception:
                band_key = None

        if band_key is None and bn:
            s = str(bn).strip()
            # if nm embedded in band name, prefer that
            m = re.search(r"(\d{3,4}(?:\.\d+)?)", s)
            if m:
                try:
                    band_key = float(m.group(1))
                except Exception:
                    band_key = None
            if band_key is None:
                band_key = s.lower()

        if band_key is None and ri is not None:
            try:
                band_key = int(ri)
            except Exception:
                band_key = None

        if band_key is None:
            band_key = Path(p).stem

        groups.setdefault(band_key, []).append(p)

    return groups, meta_cache


def _parse_dat_file(path):
    """
    Parse .dat file with key=value or key:value style (MicaSense-like).

    Returns
    -------
    dict
        Dictionary with possible keys:
        latitude, longitude, altitude, yaw, pitch, roll, band_nm.
    """
    out = {}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                elif ":" in line:
                    k, v = line.split(":", 1)
                else:
                    continue
                k_low = k.strip().lower()
                v = v.strip()

                def _to_float(s):
                    try:
                        return float(s)
                    except Exception:
                        return None

                if k_low in ("gpslatitude", "latitude", "lat"):
                    out["latitude"] = _to_float(v)
                elif k_low in ("gpslongitude", "longitude", "lon", "lng"):
                    out["longitude"] = _to_float(v)
                elif k_low in ("gpsaltitude", "altitude", "alt"):
                    out["altitude"] = _to_float(v)
                elif k_low in ("yaw", "heading"):
                    out["yaw"] = _to_float(v)
                elif k_low == "pitch":
                    out["pitch"] = _to_float(v)
                elif k_low == "roll":
                    out["roll"] = _to_float(v)
                elif "wavelength" in k_low or k_low.endswith("nm"):
                    nm = _to_float(v)
                    if nm:
                        out["band_nm"] = nm
    except Exception:
        pass
    return out


def _get_exif_position(path):
    """
    Simple fallback: try to read GPS EXIF tags via rasterio.

    Returns
    -------
    dict
        May contain latitude, longitude, altitude.
    """
    meta = {}
    try:
        with rasterio.open(path) as ds:
            tags = ds.tags() or {}
            for k in ("GPSLatitude", "GPSLongitude", "GPSAltitude"):
                v = tags.get(k)
                if v is None:
                    continue
                try:
                    vf = float(str(v).strip())
                except Exception:
                    vf = None
                if k == "GPSLatitude" and vf is not None:
                    meta["latitude"] = vf
                if k == "GPSLongitude" and vf is not None:
                    meta["longitude"] = vf
                if k == "GPSAltitude" and vf is not None:
                    meta["altitude"] = vf
    except Exception:
        pass
    return meta


def build_geodata_for_images(image_paths):
    """
    Combine .dat metadata and EXIF GPS tags per image (lon/lat/alt/yaw/pitch/roll).

    Parameters
    ----------
    image_paths : list of str
        Paths to the input images.

    Returns
    -------
    dict
        Mapping image_path -> dict with geodata.
    """
    out = {}
    for p in image_paths:
        base, _ = os.path.splitext(p)
        cand_dat = base + ".dat"
        dat = _parse_dat_file(cand_dat) if os.path.exists(cand_dat) else {}
        exif = _get_exif_position(p)
        # Also scan TIFF metadata for band/wavelength and DLS/panel hints
        tmeta = {}
        try:
            tmeta = scan_tiff_metadata(p) or {}
        except Exception:
            tmeta = {}
        geo = {}
        for k in ("latitude", "longitude", "altitude", "yaw", "pitch", "roll"):
            if k in dat and dat[k] is not None:
                geo[k] = dat[k]
            elif k in exif and exif[k] is not None:
                geo[k] = exif[k]

        # Store spectral & radiometric hints (non-breaking extras)
        if tmeta.get("central_wavelength_nm") is not None:
            geo["band_nm"] = float(tmeta["central_wavelength_nm"])
        if tmeta.get("band_name"):
            geo["band_name"] = str(tmeta["band_name"])
        if tmeta.get("dls"):
            geo["dls"] = dict(tmeta["dls"])
        out[p] = geo
    return out

    # --------------------------------------------------------------------------
    # Automatic band metadata inference
    # --------------------------------------------------------------------------
    # Helper functions below enable automatic detection of band wavelengths and
    # human‑readable labels.  They search TIFF metadata, sidecar files and
    # camera presets so that the user no longer needs to manually enter band
    # information.  See the accompanying documentation and research for details
    # about supported cameras and metadata keys.

    # Keys commonly used to store wavelength information in GeoTIFF metadata.  Both
    # uppercase and lowercase versions are handled when scanning tags.
    _WL_KEYS = (
        "WAVELENGTH",
        "wavelength",
        "CENTER_WAVELENGTH",
        "center_wavelength",
        "CENTRAL_WAVELENGTH",
        "central_wavelength",
        "CentralWavelength",
        "CenterWavelength",
        "BANDNAME",
        "BandName",
        "BAND_NAME",
        "band_name",
        "DESCRIPTION",
        "description",
    )

    # Approximate spectral names used when labelling bands.  Each entry is a
    # reference wavelength (nm) and the corresponding label.  During
    # inference, the nearest reference within a tolerance will be used.
    _SPECTRAL_NAMES = [
        (450, "Blue"),
        (470, "Blue"),
        (475, "Blue"),
        (500, "Green"),
        (540, "Green"),
        (560, "Green"),
        (630, "Red"),
        (650, "Red"),
        (660, "Red"),
        (668, "Red"),
        (705, "RedEdge"),
        (717, "RedEdge"),
        (730, "RedEdge"),
        (735, "RedEdge"),
        (760, "RedEdge"),
        (790, "NIR"),
        (800, "NIR"),
        (810, "NIR"),
        (840, "NIR"),
        (842, "NIR"),
        (860, "NIR"),
        (900, "NIR"),
        (940, "NIR"),
        (950, "NIR"),
        (970, "NIR"),
        (1000, "NIR"),
        (1100, "SWIR"),
    ]

    def _parse_first_float(text):
        """
        Extract the first floating point number from a string.

        Parameters
        ----------
        text : str or any
            Input text to scan.

        Returns
        -------
        float or None
            The first detected float value, or None if none was found.
        """
        if text is None:
            return None
        s = str(text)
        m = re.search(r"(-?\d+(?:\.\d+)?)", s)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    def _infer_name_from_nm(nm, tol=30.0):
        """
        Infer a human‑readable spectral band name from a wavelength.

        Parameters
        ----------
        nm : float
            Wavelength in nanometres.
        tol : float, optional
            Maximum allowed difference (in nm) to match a reference (default 30).

        Returns
        -------
        str or None
            The inferred spectral name (e.g. 'Red', 'Green'), or None if no
            reference is within the tolerance.
        """
        if nm is None:
            return None
        best_name = None
        best_diff = float("inf")
        for ref_nm, name in _SPECTRAL_NAMES:
            diff = abs(float(nm) - float(ref_nm))
            if diff < best_diff:
                best_name = name
                best_diff = diff
        return best_name if best_diff <= tol else None

    def _extract_nm_from_tif_metadata(tif_path):
        """
        Extract a wavelength (nm) from a GeoTIFF's internal metadata.  This
        function checks band descriptions, per‑band tags and dataset tags for
        numeric values in the valid range (350–2500 nm).

        Parameters
        ----------
        tif_path : str
            Path to the GeoTIFF file.

        Returns
        -------
        float or None
            Detected wavelength, or None if none could be extracted.
        """
        try:
            with rasterio.open(tif_path) as ds:
                # 1) Band descriptions
                try:
                    if ds.descriptions:
                        desc = ds.descriptions[0]
                        nm = _parse_first_float(desc)
                        if nm and 350 <= nm <= 2500:
                            return nm
                except Exception:
                    pass
                # 2) Tags (band and dataset)
                for tags in (ds.tags(1) or {}, ds.tags() or {}):
                    # Direct lookup by common keys
                    for k in _WL_KEYS:
                        if k in tags:
                            nm_val = _parse_first_float(tags.get(k))
                            if nm_val and 350 <= nm_val <= 2500:
                                return nm_val
                    # Fallback: scan all tag values for a numeric value
                    for v in tags.values():
                        nm_val = _parse_first_float(v)
                        if nm_val and 350 <= nm_val <= 2500:
                            return nm_val
        except Exception as e:
            meta["raw_tags"]["_rasterio_error"] = repr(e)
            pass
        return None

    def _extract_nm_from_sidecar(tif_path):
        """
        Attempt to read a wavelength (nm) from sidecar files (.xmp, .xml, .json,
        .txt) associated with a GeoTIFF.  Looks for keywords such as
        'CentralWavelength', 'CenterWavelength' or 'wavelength' and extracts the
        first numeric value between 350 and 2500 nm.

        Parameters
        ----------
        tif_path : str
            Path to the GeoTIFF file.

        Returns
        -------
        float or None
            Detected wavelength, or None if none could be extracted.
        """
        base, _ = os.path.splitext(tif_path)
        side_exts = [
            ".xmp",
            ".XMP",
            ".xml",
            ".XML",
            ".json",
            ".JSON",
            ".txt",
            ".TXT",
        ]
        patterns = [
            r"CentralWavelength\s*[<:=]?\s*(\d+(?:\.\d+)?)",
            r"CenterWavelength\s*[<:=]?\s*(\d+(?:\.\d+)?)",
            r"central_wavelength\s*[<:=]?\s*(\d+(?:\.\d+)?)",
            r"center_wavelength\s*[<:=]?\s*(\d+(?:\.\d+)?)",
            r"Wavelength\s*[<:=]?\s*(\d+(?:\.\d+)?)",
            r"wavelength\s*[<:=]?\s*(\d+(?:\.\d+)?)",
        ]
        for ext in side_exts:
            sc_path = base + ext
            if not os.path.exists(sc_path):
                continue
            try:
                with open(sc_path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
            except Exception:
                continue
            # Keyword search
            for pat in patterns:
                m = re.search(pat, txt, flags=re.IGNORECASE)
                if m:
                    try:
                        nm_val = float(m.group(1))
                        if 350 <= nm_val <= 2500:
                            return nm_val
                    except Exception:
                        pass
            # Check for number followed by 'nm'
            m2 = re.search(r"(\d+(?:\.\d+)?)\s*nm", txt, flags=re.IGNORECASE)
            if m2:
                try:
                    nm_val = float(m2.group(1))
                    if 350 <= nm_val <= 2500:
                        return nm_val
                except Exception:
                    pass
            # Fallback: any numeric value
            m3 = re.search(r"(\d+(?:\.\d+)?)", txt)
            if m3:
                try:
                    nm_val = float(m3.group(1))
                    if 350 <= nm_val <= 2500:
                        return nm_val
                except Exception:
                    pass
        return None

    def _extract_camera_model(tif_path):
        """
        Detect the camera model from TIFF and sidecar metadata.  Aids in
        selecting a preset mapping when explicit wavelength information is
        missing.

        Parameters
        ----------
        tif_path : str
            Path to a sample GeoTIFF file.

        Returns
        -------
        str or None
            A string identifying a known camera model, or None if detection
            fails.
        """
        candidates = []
        # Scan TIFF dataset tags for camera/model information
        try:
            with rasterio.open(tif_path) as ds:
                tags = ds.tags() or {}
                for k, v in tags.items():
                    key = k.lower()
                    if key in (
                        "model",
                        "make",
                        "cameramodelname",
                        "unique_cameramodel",
                        "rigname",
                        "camera",
                        "camera_model",
                        "camera-model",
                        "model_name",
                    ):
                        if v:
                            candidates.append(str(v))
        except Exception as e:
            meta["raw_tags"]["_rasterio_error"] = repr(e)
            pass
        # Read sidecar snippets for camera hints
        base, _ = os.path.splitext(tif_path)
        for ext in (".xmp", ".XMP", ".xml", ".XML", ".json", ".JSON", ".txt", ".TXT"):
            sc_path = base + ext
            if not os.path.exists(sc_path):
                continue
            try:
                with open(sc_path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read(50000)
                candidates.append(txt[:5000])
            except Exception:
                continue
        text = "|".join(candidates).lower()
        if not text:
            return None
        if "mavic" in text and "3" in text:
            return "DJI Mavic 3 Multispectral"
        if ("p4" in text or "phantom" in text) and ("multi" in text or "spectral" in text):
            return "DJI P4 Multispectral"
        if "sequoia" in text:
            return "Parrot Sequoia"
        if "sentera" in text and "6" in text:
            return "Sentera 6X"
        if "altum" in text:
            return "MicaSense Altum"
        if "rededge" in text or "red edge" in text:
            # Differentiate RedEdge‑P vs RedEdge‑M when possible
            if "p" in text:
                return "MicaSense RedEdge-P"
            return "MicaSense RedEdge-M"
        return None

# -------------------------
# ODM (Docker) — canonical command
# -------------------------
def _build_odm_command(
    project_dir_on_host,
    dataset_name="dataset",
    ortho_resolution_cm=5.0,
    fast=True,
    feature_quality="medium",
    max_concurrency=0,
):
    """
    Build the ODM Docker command for a given project directory.
    """
    cmd = [
        "docker",
        "run",
        "--rm",
        "-t",
        "-v",
        f"{project_dir_on_host}:/datasets",
        "opendronemap/odm",
        "--project-path",
        "/datasets",
        dataset_name,
        "--orthophoto-resolution",
        str(float(ortho_resolution_cm)),
        "--feature-quality",
        str(feature_quality),
        "--skip-report",
        "--force-gps",
    ]
    if fast:
        cmd += ["--fast-orthophoto", "--skip-3dmodel"]
    if max_concurrency and int(max_concurrency) > 0:
        cmd += ["--max-concurrency", str(int(max_concurrency))]
    return cmd


def _write_geo_txt(project_dir, image_paths, geodata, projection="EPSG:4326"):
    """
    Create ODM-compatible geo.txt file.

    Parameters
    ----------
    project_dir : str
        ODM project directory.
    image_paths : list of str
        List of image paths.
    geodata : dict
        Mapping path -> dict with latitude/longitude/altitude/yaw/pitch/roll.
    projection : str, optional
        CRS identifier for ODM (default: EPSG:4326).
    """
    geo_path = os.path.join(project_dir, "geo.txt")
    with open(geo_path, "w", encoding="utf-8") as f:
        f.write(projection + "\n")
        for p in image_paths:
            base = os.path.basename(p)
            g = geodata.get(p, {})
            lon = g.get("longitude")
            lat = g.get("latitude")
            alt = g.get("altitude")
            yaw = g.get("yaw")
            pitch = g.get("pitch")
            roll = g.get("roll")
            # ODM skips lines without lon/lat
            if lon is None or lat is None:
                continue
            line = [base, str(lon), str(lat)]
            if alt is not None:
                line.append(str(alt))
            if yaw is not None:
                line.append(str(yaw))
            if pitch is not None:
                line.append(str(pitch))
            if roll is not None:
                line.append(str(roll))
            f.write(" ".join(line) + "\n")
    return geo_path


# -------------------------
# Band -> wavelength mapping (UI with presets)
# -------------------------
_PRESETS = {
    # MicaSense sensors
    "MicaSense RedEdge-M": {"1": 475, "2": 560, "3": 668, "4": 717, "5": 840},
    "MicaSense RedEdge-P": {"1": 475, "2": 560, "3": 668, "4": 717, "5": 842},
    "MicaSense Altum": {
        "1": 475,
        "2": 560,
        "3": 668,
        "4": 717,
        "5": 842,
        "6": 1100,
    },
    # DJI multispectral sensors
    # Phantom 4 Multispectral: Blue~450 nm, Green~560 nm, Red~650 nm,
    # RedEdge~730 nm, NIR~840 nm【660939532244122†L191-L204】.
    "DJI P4 Multispectral": {"1": 450, "2": 560, "3": 650, "4": 730, "5": 840},
    # Mavic 3 multispectral bands: Green 560 nm, Red 650 nm, RedEdge 730 nm,
    # NIR 860 nm【468661628561316†L66-L76】.
    "DJI Mavic 3 Multispectral": {"1": 560, "2": 650, "3": 730, "4": 860},
    # Sentera sensors (6X) central wavelengths【543803204923396†L205-L210】.
    "Sentera 6X": {"1": 475, "2": 550, "3": 670, "4": 715, "5": 840},
    # Parrot Sequoia multispectral bands: Green 550 nm, Red 660 nm,
    # RedEdge 735 nm, NIR 790 nm【120072669245677†L2007-L2013】.
    "Parrot Sequoia": {"1": 550, "2": 660, "3": 735, "4": 790},
}


def map_bands_to_nm_ui(band_keys):
    """
    UI dialog to map band index -> wavelength (nm), with sensor presets
    and manual editing.

    Parameters
    ----------
    band_keys : list of str
        Band identifiers as parsed from filenames.

    Returns
    -------
    dict or None
        Mapping band_key -> wavelength_nm, or None if cancelled.
    """
    win = Toplevel()
    win.title("Map bands to wavelengths (nm)")
    win.geometry("420x360")
    win.grab_set()

    Label(win, text="Sensor preset (optional):").pack(pady=(10, 4))
    preset_var = StringVar(value="(None)")
    presets = ["(None)"] + list(_PRESETS.keys())
    combo = ttk.Combobox(
        win,
        values=presets,
        textvariable=preset_var,
        state="readonly",
    )
    combo.current(0)
    combo.pack(padx=10, fill="x")

    Label(win, text="Edit/confirm wavelength (nm) per band:").pack(pady=(12, 4))

    entries = {}
    frm = ttk.Frame(win)
    frm.pack(fill="both", expand=True, padx=10)
    for k in sorted(band_keys, key=lambda x: (len(x), x)):
        row = ttk.Frame(frm)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=f"Band {k}", width=10).pack(side="left")
        v = StringVar(value="")
        ent = ttk.Entry(row, textvariable=v, width=10)
        ent.pack(side="left")
        entries[k] = v

    def apply_preset(*_):
        name = preset_var.get()
        if name in _PRESETS:
            for k, val in _PRESETS[name].items():
                if k in entries:
                    entries[k].set(str(val))

    combo.bind("<<ComboboxSelected>>", apply_preset)

    result = {"ok": False, "out": {}}

    def confirmar():
        out = {}
        try:
            for k, var in entries.items():
                txt = var.get().strip()
                if not txt:
                    raise ValueError(f"Band {k} has no value.")
                out[k] = float(txt)
            result["ok"] = True
            result["out"] = out
            win.destroy()
        except Exception as e:
            _error(f"Invalid values: {e}")

    Button(win, text="Confirm", command=confirmar).pack(pady=10)
    win.wait_window()
    return result["out"] if result["ok"] else None


# -------------------------
# Alignment helpers (no georef): phase + ECC (affine)
# -------------------------
def _normalize01(img):
    vmin = np.nanmin(img)
    vmax = np.nanmax(img)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return img.astype(np.float32)
    return ((img - vmin) / (vmax - vmin + 1e-12)).astype(np.float32)


def _estimate_translation_phase(ref_f32, mov_f32):
    """
    Estimate translation using phase correlation with Hanning windowing.
    """
    win = np.outer(np.hanning(ref_f32.shape[0]), np.hanning(ref_f32.shape[1])).astype(
        np.float32
    )
    r = cv2.normalize(ref_f32, None, 0, 1, cv2.NORM_MINMAX) * win
    m = cv2.normalize(mov_f32, None, 0, 1, cv2.NORM_MINMAX) * win
    (shift_y, shift_x), _ = cv2.phaseCorrelate(r, m)
    return float(shift_x), float(shift_y)


def _align_to_template_image(ref, mov):
    """
    Align mov → ref using:
      1) Phase correlation (translation);
      2) ECC (affine) in pyramid to stabilize;
    Returns the aligned image as float32.
    """
    H, W = ref.shape
    ref_n = _normalize01(ref.astype(np.float32))
    mov_n = _normalize01(mov.astype(np.float32))

    # (1) Translation via phase correlation
    tx, ty = 0.0, 0.0
    try:
        tx, ty = _estimate_translation_phase(ref_n, mov_n)
    except Exception:
        pass
    warp = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)

    # (2) ECC affine refinement
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        scale = max(H, W) / 512.0
        if scale > 1.0:
            small_ref = cv2.resize(
                ref_n, (int(W / scale), int(H / scale)), interpolation=cv2.INTER_AREA
            )
            small_mov = cv2.resize(
                mov_n, (int(W / scale), int(H / scale)), interpolation=cv2.INTER_AREA
            )
            wm_small = warp.copy()
            wm_small[0, 2] /= scale
            wm_small[1, 2] /= scale
            cv2.findTransformECC(
                small_ref,
                small_mov,
                wm_small,
                cv2.MOTION_AFFINE,
                criteria,
                None,
                5,
            )
            warp = wm_small.copy()
            warp[0, 2] *= scale
            warp[1, 2] *= scale
        else:
            cv2.findTransformECC(
                ref_n,
                mov_n,
                warp,
                cv2.MOTION_AFFINE,
                criteria,
                None,
                5,
            )

        aligned = cv2.warpAffine(
            mov.astype(np.float32),
            warp,
            (W, H),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        ).astype(np.float32)
        return aligned
    except Exception:
        # Fallback: translation only
        try:
            aligned = cv2.warpAffine(
                mov.astype(np.float32),
                warp,
                (W, H),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            ).astype(np.float32)
            return aligned
        except Exception:
            return mov.astype(np.float32)


# -------------------------
# Stack orthos to the same grid — with image-based alignment fallback
# -------------------------
def _reproject_to_template(src_path, template_ds, resampling=Resampling.nearest):
    """Reproject src_path to the grid/CRS of template_ds."""
    with rasterio.open(src_path) as src:
        dst = np.zeros(
            (template_ds.height, template_ds.width),
            dtype="float32",
        )
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_ds.transform,
            dst_crs=template_ds.crs,
            resampling=resampling,
        )
        return dst


def stack_orthos_same_grid(ortho_paths):
    """
    Stack multiple orthomosaics onto the grid of the first band.

    Strategy:
      - If all orthos have the same CRS/transform/size → read directly;
      - If CRS matches but transform/size differ → reproject to first band's grid;
      - If CRS is missing or unusable → use image-based alignment
        (phase + ECC) on first band's grid.

    Parameters
    ----------
    ortho_paths : dict
        Mapping band_key -> ortho GeoTIFF path.

    Returns
    -------
    cube : np.ndarray
        Array (H, W, B) with stacked bands.
    transform : affine.Affine
        Spatial transform of the target grid.
    crs : rasterio.crs.CRS or None
        CRS of the resulting stack.
    """
    # Robust key ordering: if keys are numeric wavelengths, sort numerically; else fallback to string ordering
    def _sort_key(_k):
        try:
            # numeric wavelength keys (int/float) or numeric strings
            if isinstance(_k, (int, float)):
                return (0, float(_k))
            if isinstance(_k, str):
                s = _k.strip()
                try:
                    return (0, float(s))
                except Exception:
                    return (1, len(s), s)
            s = str(_k)
            return (1, len(s), s)
        except Exception:
            return (2, 0, '')
    keys = sorted(ortho_paths.keys(), key=_sort_key)
    first = ortho_paths[keys[0]]
    with rasterio.open(first) as temp:
        transform, crs = temp.transform, temp.crs
        H, W = temp.height, temp.width
        ref_arr = temp.read(1).astype(np.float32)

    bands = [ref_arr]
    for k in keys[1:]:
        p = ortho_paths[k]
        try:
            with rasterio.open(p) as ds:
                same_grid = (
                    crs is not None
                    and ds.crs == crs
                    and ds.transform == transform
                    and ds.width == W
                    and ds.height == H
                )
                if same_grid:
                    arr = ds.read(1).astype(np.float32)
                elif crs is not None and ds.crs is not None:
                    # Reproject to the first band's grid
                    with rasterio.open(first) as tmpl:
                        arr = _reproject_to_template(
                            p,
                            tmpl,
                            resampling=Resampling.nearest,
                        )
                else:
                    # No reliable CRS: align by image on first band's grid
                    mov = ds.read(1).astype(np.float32)
                    if mov.shape != (H, W):
                        mov = cv2.resize(
                            mov,
                            (W, H),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    arr = _align_to_template_image(ref_arr, mov)
        except Exception:
            # Full fallback: open and resize + align
            try:
                with rasterio.open(p) as ds2:
                    mov = ds2.read(1).astype(np.float32)
            except Exception:
                mov = cv2.imread(p, cv2.IMREAD_UNCHANGED).astype(np.float32)
            if mov.ndim == 3:
                mov = mov[..., 0]
            if mov.shape != (H, W):
                mov = cv2.resize(mov, (W, H), interpolation=cv2.INTER_NEAREST)
            arr = _align_to_template_image(ref_arr, mov)

        bands.append(arr)

    cube = np.stack(bands, axis=-1).astype(np.float32)
    return cube, transform, crs


# -------------------------
# Save products (multiband + per-band + metadata)
# -------------------------
def salvar_produtos(path_base, cube, wavelengths, transform, crs, dtype="float32"):
    """
    Save multiband GeoTIFF + .npy + .json with per-band metadata.

    Per-band:
      - DESCRIPTION: 'Band i - NNN nm'
      - Tag: WAVELENGTH = NNN (float)
    Dataset-level:
      - BAND_DESCRIPTIONS (pipe-separated)

    Returns
    -------
    dict
        Dictionary with keys 'tif', 'npy', 'json' pointing to each output.
    """
    import rasterio
    import numpy as np
    import json

    h, w, b = cube.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": b,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 3,
    }
    wavelengths = list(wavelengths or [])
    if len(wavelengths) != b:
        wavelengths = (wavelengths + [None] * b)[:b]

    tif_path = f"{path_base}.tif"
    with rasterio.open(tif_path, "w", **profile) as dst:
        descs = []
        for i in range(b):
            dst.write(cube[:, :, i].astype(dtype), i + 1)
            nm = wavelengths[i]
            # Build a more descriptive label when possible.  Use the spectral
            # name (e.g. Red, NIR) if `_infer_name_from_nm` recognises the
            # wavelength; otherwise fall back to a generic 'Band i - NNN nm'.
            if nm is None:
                desc = f"Band {i + 1}"
            else:
                name = _infer_name_from_nm(nm)
                if name:
                    desc = f"{name} - {int(round(float(nm)))} nm"
                else:
                    desc = f"Band {i + 1} - {int(round(float(nm)))} nm"
            descs.append(desc)
            # band description and tags
            try:
                dst.set_band_description(i + 1, desc)
            except Exception:
                pass
            try:
                dst.update_tags(bidx=i + 1, DESCRIPTION=desc)
            except Exception:
                pass
            if nm is not None:
                val = str(float(nm))
                try:
                    dst.update_tags(bidx=i + 1, WAVELENGTH=val)
                except Exception:
                    try:
                        dst.update_tags(bidx=i + 1, wavelength=val)
                    except Exception:
                        pass
        # Dataset-level tags
        try:
            dst.update_tags(
                BAND_COUNT=str(b),
                BAND_DESCRIPTIONS="|".join(descs),
            )
        except Exception as e:
            meta["raw_tags"]["_rasterio_error"] = repr(e)
            pass

    np.save(f"{path_base}.npy", cube.astype(np.float32))
    with open(f"{path_base}.json", "w") as f:
        json.dump(
            {
                "wavelengths": [
                    None if v is None else float(v) for v in wavelengths
                ]
            },
            f,
            indent=2,
        )
    return {
        "tif": tif_path,
        "npy": f"{path_base}.npy",
        "json": f"{path_base}.json",
    }


def salvar_bandas_individuais(path_base, cube, wavelengths, transform, crs, dtype="float32"):
    """
    Save one GeoTIFF per band, with wavelength in file name (when available).
    """
    import rasterio

    out_paths = []
    h, w, b = cube.shape
    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": 1,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "predictor": 3,
    }
    wavelengths = list(wavelengths or [])
    if len(wavelengths) != b:
        wavelengths = (wavelengths + [None] * b)[:b]

    for i, nm in enumerate(wavelengths, start=1):
        nm_suf = f"_{int(round(float(nm)))}nm" if nm is not None else ""
        out_path = f"{path_base}_band{i}{nm_suf}.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            # Write the single band data
            dst.write(cube[:, :, i - 1].astype(dtype), 1)
            # Use spectral name when possible for single‑band output
            if nm is None:
                desc = f"Band {i}"
            else:
                name = _infer_name_from_nm(nm)
                if name:
                    desc = f"{name} - {int(round(float(nm)))} nm"
                else:
                    desc = f"Band {i} - {int(round(float(nm)))} nm"
            # Set band description and metadata
            try:
                dst.set_band_description(1, desc)
            except Exception:
                pass
            try:
                dst.update_tags(bidx=1, DESCRIPTION=desc)
            except Exception:
                pass
            if nm is not None:
                val = str(float(nm))
                try:
                    dst.update_tags(bidx=1, WAVELENGTH=val)
                except Exception:
                    try:
                        dst.update_tags(bidx=1, wavelength=val)
                    except Exception:
                        pass
        out_paths.append(out_path)
    return out_paths


# -------------------------
# Explicit folder selector — UX
# -------------------------
def _ask_images_folder():
    """
    Ask the user to select the folder containing the flight images (.tif/.tiff).
    Outputs will be saved into a '__odm_outputs' subfolder.
    """
    _info(
        "STEP 1/3 — Select the folder containing the FLIGHT IMAGES.\n\n"
        "• It must contain the .tif/.tiff photos (and optionally .dat files).\n"
        "• Results will be created in a '__odm_outputs' subfolder under this folder.",
        title="Select IMAGE folder",
    )
    while True:
        folder = filedialog.askdirectory(
            title="Select the FOLDER with the flight images (.tif/.tiff)"
        )
        if not folder:
            return None
        has_tif = False
        for root, _, files in os.walk(folder):
            if any(fn.lower().endswith((".tif", ".tiff")) for fn in files):
                has_tif = True
                break
        if not has_tif:
            _warn(
                "The selected folder does not contain any .tif/.tiff files.\n\n"
                "Please select the correct folder containing the flight images."
            )
            continue
        return folder


# -------------------------
# Processing options dialog (FAST / FULL)
# -------------------------
def _ask_processing_options():
    """
    Ask the user for ODM processing options: fast/full, resolution, quality, etc.
    """
    win = Toplevel()
    win.title("STEP 2/3 — Processing options")
    win.geometry("520x320")
    win.grab_set()

    Label(
        win,
        text="Choose the profile and adjust options if needed:",
        font=("Arial", 11, "bold"),
    ).pack(pady=(10, 6))

    fast_var = tk.IntVar(value=1)
    tk.Checkbutton(
        win,
        text="FAST mode (fast-orthophoto and skip 3D/mesh)",
        variable=fast_var,
    ).pack(anchor="w", padx=12)

    Label(win, text="Orthophoto resolution (cm/pixel):").pack(
        anchor="w", padx=12, pady=(10, 0)
    )
    res_var = tk.StringVar(value="10")
    ttk.Entry(win, textvariable=res_var, width=8).pack(
        anchor="w",
        padx=12,
        pady=(2, 8),
    )

    Label(win, text="Feature quality:").pack(anchor="w", padx=12, pady=(2, 0))
    q_var = StringVar(value="medium")
    ttk.Combobox(
        win,
        values=["ultra", "high", "medium", "low", "lowest"],
        textvariable=q_var,
        state="readonly",
        width=10,
    ).pack(anchor="w", padx=12, pady=(2, 6))

    Label(win, text="Limit threads (optional):").pack(
        anchor="w", padx=12, pady=(2, 0)
    )
    mc_var = tk.StringVar(value="0")
    ttk.Entry(win, textvariable=mc_var, width=8).pack(
        anchor="w",
        padx=12,
        pady=(2, 8),
    )

    save_ind_var = tk.IntVar(value=0)
    tk.Checkbutton(
        win,
        text="Also save one GeoTIFF per band (slower, useful for QGIS)",
        variable=save_ind_var,
    ).pack(anchor="w", padx=12, pady=(2, 0))

    # Preset buttons
    frm_presets = ttk.Frame(win)
    frm_presets.pack(anchor="w", padx=12, pady=(10, 6))
    ttk.Label(frm_presets, text="Presets:").grid(row=0, column=0, padx=(0, 8))

    def set_preset_fast10(*_):
        fast_var.set(1)
        res_var.set("10")
        q_var.set("medium")
        mc_var.set("0")
        save_ind_var.set(0)

    def set_preset_fast5(*_):
        fast_var.set(1)
        res_var.set("5")
        q_var.set("medium")
        mc_var.set("0")
        save_ind_var.set(0)

    def set_preset_full5(*_):
        fast_var.set(0)
        res_var.set("5")
        q_var.set("high")
        mc_var.set("0")
        save_ind_var.set(1)

    ttk.Button(frm_presets, text="Fast (10 cm)", command=set_preset_fast10).grid(
        row=0, column=1, padx=4
    )
    ttk.Button(frm_presets, text="Balanced (5 cm)", command=set_preset_fast5).grid(
        row=0, column=2, padx=4
    )
    ttk.Button(frm_presets, text="Full (5 cm)", command=set_preset_full5).grid(
        row=0, column=3, padx=4
    )

    result = {"ok": False}

    def _ok():
        result["ok"] = True
        win.destroy()

    Button(win, text="Start processing", command=_ok).pack(pady=12)
    win.wait_window()

    if not result["ok"]:
        return None
    try:
        res_cm = float(res_var.get())
        res_cm = res_cm if res_cm > 0 else 10.0
    except Exception:
        res_cm = 10.0
    try:
        max_conc = int(mc_var.get())
        max_conc = max(0, max_conc)
    except Exception:
        max_conc = 0

    return {
        "fast": bool(fast_var.get()),
        "res_cm": float(res_cm),
        "feat_quality": q_var.get(),
        "max_conc": int(max_conc),
        "save_individual": bool(save_ind_var.get()),
    }


# -------------------------
# Run ODM per band (logging and robustness)
# -------------------------
def run_odm_per_band(
    groups,
    out_root,
    res_cm=10.0,
    fast=True,
    feat_quality="medium",
    max_conc=0,
):
    """
    Run ODM once per band group, creating one project per band.

    Parameters
    ----------
    groups : dict
        band_key -> [image_paths...]
    out_root : str
        Root folder where ODM band projects will be created.
    """
    if not _docker_available():
        _error(
            "Docker was not found.\n\n"
            "Please install Docker Desktop and run:\n"
            "  docker pull opendronemap/odm"
        )
        return {}

    os.makedirs(out_root, exist_ok=True)
    ortho_paths = {}

    for band_idx, paths in sorted(groups.items(), key=lambda kv: kv[0]):
        if not paths:
            continue

        proj_dir = os.path.join(out_root, f"odm_band_{band_idx}")
        os.makedirs(proj_dir, exist_ok=True)

        dataset_name = "dataset"
        dataset_dir = os.path.join(proj_dir, dataset_name)
        images_dir = os.path.join(dataset_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        # Link/copy images into dataset/images
        for p in paths:
            base = os.path.basename(p)
            dst = os.path.join(images_dir, base)
            if not os.path.exists(dst):
                try:
                    os.link(p, dst)
                except Exception:
                    import shutil

                    shutil.copy2(p, dst)

        # Per-image geodata and geo.txt
        geodata = build_geodata_for_images(paths)
        _write_geo_txt(proj_dir, paths, geodata, projection="EPSG:4326")

        cmd = _build_odm_command(
            proj_dir,
            dataset_name=dataset_name,
            ortho_resolution_cm=res_cm,
            fast=fast,
            feature_quality=feat_quality,
            max_concurrency=max_conc,
        )
        log_path = os.path.join(proj_dir, "odm.log")
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(">>> CMD:\n" + " ".join(cmd) + "\n\n")
                f.write(">>> STDOUT:\n" + (proc.stdout or "") + "\n\n")
                f.write(">>> STDERR:\n" + (proc.stderr or "") + "\n")

            if proc.returncode != 0:
                _warn(
                    f"ODM returned an error for band {band_idx} (code {proc.returncode}).\n"
                    f"See log:\n{log_path}"
                )
                continue

            cand = os.path.join(
                dataset_dir,
                "odm_orthophoto",
                "odm_orthophoto.tif",
            )
            if os.path.exists(cand):
                ortho_paths[band_idx] = cand
            else:
                _warn(
                    f"Could not find ODM orthophoto for band {band_idx}.\n"
                    f"Check the log:\n{log_path}"
                )
        except Exception as e:
            _error(
                f"Failed to run ODM (band {band_idx}): {e}\n\n"
                f"Check the log (if it exists):\n{log_path}"
            )

    return ortho_paths


# -------------------------
# Main GeoImport wizard GUI
# -------------------------
def geoimport_wizard_gui():
    """
    Main GeoImport wizard for EasySpectra.

    Workflow:
      1) Select the folder containing flight images;
      2) Choose processing profile/options (Fast/Full, resolution, etc.);
      3) Automatic output folder: <images_folder>/__odm_outputs;
      4) Run ODM per band (using geo.txt built from .dat/EXIF);
      5) Map band index -> wavelength (nm) using presets and manual edits;
      6) Stack orthomosaics to a common grid
         (reproject when possible, otherwise align by image);
      7) (Optional) Apply radiometric correction using calibration panels
         with visible ROI (ENTER/ESC);
      8) Save multiband GeoTIFF, NPY, JSON, and optionally per-band GeoTIFFs.
    """
    if not _docker_available():
        _error(
            "Docker was not found.\n\n"
            "Please install Docker Desktop and run:\n\n"
            "docker pull opendronemap/odm"
        )
        return

    images_folder = _ask_images_folder()
    if not images_folder:
        return

    # Options
    opts = _ask_processing_options()
    if not opts:
        _warn("Processing cancelled.")
        return

    res_cm = opts["res_cm"]
    fast_mode = opts["fast"]
    feat_quality = opts["feat_quality"]
    max_conc = opts["max_conc"]
    save_individual_tiffs = opts["save_individual"]

    out_root = os.path.join(images_folder, "__odm_outputs")
    os.makedirs(out_root, exist_ok=True)
    _info(
        "ODM projects and final products will be saved in:\n"
        f"{out_root}\n\n"
        "Your original image folder WILL NOT be modified."
    )

    # ------------------------------------------------------------------
    # STEP 2b) Varre metadados (TIFF) e agrupa bandas automaticamente.
    # Isso melhora MUITO o reconhecimento de bandas em cameras que nao
    # seguem padrao de nome de arquivo.
    # ------------------------------------------------------------------
    all_tifs = []
    for root, _, files in os.walk(images_folder):
        for fn in files:
            if _TIF_RE.match(fn):
                all_tifs.append(os.path.join(root, fn))
    all_tifs = sorted(all_tifs)

    if not all_tifs:
        _warn("No .tif/.tiff files were found in the selected folder.")
        return

    _info(
        "STEP 2/3 — Reading TIFF metadata (band name, wavelength, DLS/panel hints)\n"
        "and grouping images per band automatically."
    )
    groups, meta_cache = varrer_metadados_tiff(all_tifs)
    if not groups:
        _warn("Could not group TIFFs by band (no usable metadata / filenames).")
        return

    _info(
        "STEP 3/3 — Running ODM per band. This may take some time,\n"
        "especially on the first run (Docker image download)."
    )
    ortho_paths = run_odm_per_band(
        groups,
        out_root,
        res_cm=res_cm,
        fast=fast_mode,
        feat_quality=feat_quality,
        max_conc=max_conc,
    )
    if not ortho_paths:
        _error("No orthomosaics were generated by ODM.")
        return

    # 5) Band -> wavelength mapping
    # Infer wavelengths automatically from metadata/sidecars (no UI)
    nm_map, label_map = infer_nm_map_from_groups(groups, meta_cache=meta_cache)
    # Inform the user what was detected
    try:
        ordered_keys = sorted(nm_map.keys(), key=lambda x: (len(x), x))
        msg = "\n".join([
            f"{k} -> {label_map.get(k, 'Unknown')}" for k in ordered_keys
        ])
        _info(
            "Band metadata detected automatically:\n\n" + msg
        )
    except Exception:
        pass

    _info(
        "Stacking orthomosaics on a common grid.\n"
        "When possible, reprojection is used; otherwise, image-based alignment is applied."
    )
    cube, transform, crs = stack_orthos_same_grid(ortho_paths)

    # Use the same ordered_keys for stacking results (sort by length then string)
    # Robust key ordering for numeric wavelengths (float/int) and strings
    def _sort_key_odm(_k):
        try:
            if isinstance(_k, (int, float)):
                return (0, float(_k))
            if isinstance(_k, str):
                s = _k.strip()
                try:
                    return (0, float(s))
                except Exception:
                    return (1, len(s), s)
            s = str(_k)
            return (1, len(s), s)
        except Exception:
            return (2, 0, '')
    ordered_keys = sorted(ortho_paths.keys(), key=_sort_key_odm)
    # Retrieve wavelengths in the same order as bands; missing values become None
    wavelengths = [nm_map.get(k) for k in ordered_keys]

    # 6.5) OPTIONAL — Panel-based radiometric correction (visible ROI) BEFORE saving
    if messagebox.askyesno(
        "Radiometric panel correction",
        "Do you want to apply radiometric correction using white/grey/black panels\n"
        "before saving the multiband products?",
    ):
        try:
            # Late import to avoid circular dependencies;
            # function already handles panel alignment + visible ROI and ENTER/ESC logic.
            try:
                # Preferred: relative import inside the package
                from .funcoes_importacao import (
                    calibrar_cubo_por_paineis_com_bandas_gui,
                    calibrar_cubo_por_paineis_foto_unica_gui,
                    _ui_modo_calibracao_paineis,
                )
            except ImportError:
                # Fallback: absolute import (dev mode, running from root folder)
                from funcoes_importacao import (
                    calibrar_cubo_por_paineis_com_bandas_gui,
                    calibrar_cubo_por_paineis_foto_unica_gui,
                    _ui_modo_calibracao_paineis,
                )

            modo = _ui_modo_calibracao_paineis()
            if modo == "por_banda":
                cube_corr, detalhes = calibrar_cubo_por_paineis_com_bandas_gui(
                    cube, wavelengths
                )
            else:
                cube_corr, detalhes = calibrar_cubo_por_paineis_foto_unica_gui(
                    cube, wavelengths
                )
            if detalhes is not None:
                cube = cube_corr
        except Exception as e:
            _warn(
                "Panel-based correction failed:\n"
                f"{type(e).__name__}: {e}\n\n"
                "Files will be saved without radiometric correction."
            )

    # 7) Save final products
    base = simpledialog.askstring(
        "Save as",
        "Base name for saving (without extension):",
        initialvalue="multiband_mosaic",
    )
    if not base:
        return

    path_base = os.path.join(out_root, base)
    paths = salvar_produtos(path_base, cube, wavelengths, transform, crs)

    extras = ""
    if save_individual_tiffs:
        band_tifs = salvar_bandas_individuais(
            path_base,
            cube,
            wavelengths,
            transform,
            crs,
        )
        extras = "\n\nIndividual band GeoTIFFs:\n- " + "\n- ".join(band_tifs)

    _info(
        "Done!\n\n"
        f"Multiband GeoTIFF: {paths['tif']}\n"
        f"NPY array:        {paths['npy']}\n"
        f"Metadata JSON:    {paths['json']}"
        f"{extras}"
    )

