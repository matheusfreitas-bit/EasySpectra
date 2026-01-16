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

# =============== Rasterio warnings for non-georeferenced images ===============
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
# ========================================================================


# -------------------------
# UI utilities / alerts
# -------------------------
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
# .DAT parser (MicaSense) and EXIF fallback
# -------------------------
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
        geo = {}
        for k in ("latitude", "longitude", "altitude", "yaw", "pitch", "roll"):
            if k in dat and dat[k] is not None:
                geo[k] = dat[k]
            elif k in exif and exif[k] is not None:
                geo[k] = exif[k]
        out[p] = geo
    return out


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
    keys = sorted(ortho_paths.keys(), key=lambda x: (len(x), x))
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
            desc = (
                f"Band {i + 1}"
                if nm is None
                else f"Band {i + 1} - {int(round(float(nm)))} nm"
            )
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
        except Exception:
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
            dst.write(cube[:, :, i - 1].astype(dtype), 1)
            desc = (
                f"Band {i}"
                if nm is None
                else f"Band {i} - {int(round(float(nm)))} nm"
            )
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

    groups = scan_flight_folder(images_folder)
    if not groups:
        _warn("No .tif/.tiff files were found in the selected folder.")
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
    nm_map = map_bands_to_nm_ui(list(ortho_paths.keys()))
    if not nm_map:
        _warn("Wavelength mapping was cancelled.")
        return

    _info(
        "Stacking orthomosaics on a common grid.\n"
        "When possible, reprojection is used; otherwise, image-based alignment is applied."
    )
    cube, transform, crs = stack_orthos_same_grid(ortho_paths)

    ordered_keys = sorted(ortho_paths.keys(), key=lambda x: (len(x), x))
    try:
        wavelengths = [nm_map[k] for k in ordered_keys]
    except KeyError as e:
        _error(f"No wavelength was defined for band key: {e}")
        return

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

