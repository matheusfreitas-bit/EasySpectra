# radiometric_tab.py
"""
Radiometric correction tab for EasySpectra.

This tab applies per-band radiometric calibration before orthomosaic/stacking.
It supports:
  - Flight metadata recorded during capture (irradiance + sun angle when available)
  - Calibration panels (ROI-based)
  - AUTO mode (best available strategy with fallback)

Panel workflow (calibration_panels):
  1) User selects a folder (or files) containing panel TIFFs.
  2) Bands are detected from TIFF metadata (same metadata mining path used by the pipeline).
  3) User selects ONE ROI per band (interactive rectangle selection).
  4) User provides reflectance as:
       - a single global value (fallback), and/or
       - per-band values.
  5) The correction + saving remains identical to the AUTO pipeline path (backend writer).
"""

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

from .radiometric_preprocess import RadiometricConfig, apply_radiometric_corrections
from .radiometric_preprocess import scan_tiff_metadata

# Try to import the same band-key function used by the backend.
try:
    from .radiometric_preprocess import _band_key_from_meta  # type: ignore
except Exception:
    _band_key_from_meta = None  # fallback implemented below


def _fallback_band_key(meta: dict) -> str:
    """
    Conservative fallback if _band_key_from_meta isn't importable.
    Prefers central wavelength; otherwise band_name; otherwise filename stem.
    """
    wl = meta.get("central_wavelength_nm") or meta.get("central_wavelength") or meta.get("wavelength")
    if wl is not None:
        try:
            return f"wl_{int(round(float(wl)))}"
        except Exception:
            pass
    bn = meta.get("band_name")
    if bn:
        return f"band_{str(bn).strip().lower().replace(' ', '_')}"
    stem = os.path.splitext(os.path.basename(meta.get("path", "band")))[0]
    return f"band_{stem}"


def _band_key(meta: dict) -> str:
    if _band_key_from_meta is not None:
        try:
            return _band_key_from_meta(meta)
        except Exception:
            return _fallback_band_key(meta)
    return _fallback_band_key(meta)


def _list_tiffs(folder: str):
    out = []
    for name in os.listdir(folder):
        if name.lower().endswith((".tif", ".tiff")):
            out.append(os.path.join(folder, name))
    out.sort()
    return out


def _detect_bands_from_panel_folder(panel_folder: str):
    """
    Returns:
      band_keys: sorted unique list[str]
      example_by_band: dict[band_key] -> a representative TIFF path for ROI selection
    """
    tiffs = _list_tiffs(panel_folder)
    if not tiffs:
        return [], {}

    # scan_tiff_metadata differs across EasySpectra versions:
    # - some versions accept a LIST of TIFF paths and return a LIST of metadata dicts
    # - others accept a SINGLE TIFF path and return a single metadata dict
    try:
        metas = scan_tiff_metadata(tiffs)
        if isinstance(metas, dict):
            metas = [metas]
    except TypeError:
        metas = []
        for p in tiffs:
            m = scan_tiff_metadata(p)
            if isinstance(m, dict):
                m = dict(m)
                m.setdefault("path", p)
                metas.append(m)

    example_by_band = {}
    for meta in metas:
        bk = _band_key(meta)
        if bk not in example_by_band:
            example_by_band[bk] = meta.get("path") or meta.get("src_path") or meta.get("file") or ""
            if not example_by_band[bk]:
                # fallback to original list order
                example_by_band[bk] = tiffs[0]

    band_keys = sorted(example_by_band.keys())
    return band_keys, example_by_band


def _select_roi_matplotlib(image_2d: np.ndarray, title: str):
    """
    Interactive ROI selection via matplotlib RectangleSelector.
    Returns (x, y, w, h) in pixel coordinates, or None if cancelled.
    """
    try:
        import matplotlib
        # Prefer Tk backend for interactive selection
        matplotlib.use("TkAgg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RectangleSelector
    except Exception as e:
        messagebox.showerror(
            "ROI selection unavailable",
            f"Could not start interactive ROI selector (matplotlib/Tk backend).\n\n{e!r}"
        )
        return None

    roi = {"x0": None, "y0": None, "x1": None, "y1": None}

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.imshow(image_2d, cmap="gray")
    ax.set_axis_off()

    def onselect(eclick, erelease):
        roi["x0"], roi["y0"] = int(round(eclick.xdata)), int(round(eclick.ydata))
        roi["x1"], roi["y1"] = int(round(erelease.xdata)), int(round(erelease.ydata))

    rect = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True
    )

    # Help text
    fig.text(0.01, 0.01, "Drag to select ROI. Close window when done.", fontsize=9)

    plt.show()

    if roi["x0"] is None or roi["x1"] is None:
        return None

    x0, x1 = sorted([roi["x0"], roi["x1"]])
    y0, y1 = sorted([roi["y0"], roi["y1"]])
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    return (int(x0), int(y0), int(w), int(h))


def create_radiometric_tab(parent):
    frame = ttk.Frame(parent)

    # =============================
    # Variables
    # =============================
    input_folder_var = tk.StringVar()
    output_folder_var = tk.StringVar()

    method_var = tk.StringVar(value="auto")
    recursive_var = tk.BooleanVar(value=False)
    overwrite_var = tk.BooleanVar(value=False)

    # Optional physical correction (AUTO/AUTO+): band sensitivity
    band_sens_mode_var = tk.StringVar(value="Divide")

    # Panels: folder + ROI/reflectance state
    panel_folder_var = tk.StringVar()  # single or before when before/after enabled
    panel_before_after_var = tk.BooleanVar(value=False)
    panel_folder_after_var = tk.StringVar()
    panel_reflectance_global_var = tk.DoubleVar(value=0.60)

    # Internal state containers (filled by interactive dialogs)
    panel_band_keys = []
    panel_example_by_band = {}
    panel_roi_by_band = {}           # band_key -> (x,y,w,h) (single)
    panel_roi_by_band_before = {}    # band_key -> (x,y,w,h)
    panel_roi_by_band_after = {}     # band_key -> (x,y,w,h)
    panel_reflectance_by_band = {}   # band_key -> float

    status_var = tk.StringVar(value="Ready.")

    # =============================
    # UI helpers
    # =============================
    def browse_input():
        path = filedialog.askdirectory()
        if path:
            input_folder_var.set(path)

    def browse_output():
        path = filedialog.askdirectory()
        if path:
            output_folder_var.set(path)

    def browse_panel_folder():
        path = filedialog.askdirectory()
        if path:
            panel_folder_var.set(path)
            _refresh_panel_bands()

    def browse_panel_folder_after():
        path = filedialog.askdirectory()
        if path:
            panel_folder_after_var.set(path)
            _refresh_panel_bands()

    def _refresh_panel_bands():
        nonlocal panel_band_keys, panel_example_by_band
        use_ba = bool(panel_before_after_var.get())

        if not use_ba:
            folder = panel_folder_var.get().strip()
            if not folder:
                panel_band_keys = []
                panel_example_by_band = {}
                return
            if not os.path.isdir(folder):
                messagebox.showerror("Panels", "Panel folder does not exist.")
                return
            band_keys, example_by_band = _detect_bands_from_panel_folder(folder)
            panel_band_keys = band_keys
            panel_example_by_band = example_by_band

            # Keep existing ROIs/reflectances where possible
            for k in list(panel_roi_by_band.keys()):
                if k not in panel_band_keys:
                    panel_roi_by_band.pop(k, None)
            for k in list(panel_reflectance_by_band.keys()):
                if k not in panel_band_keys:
                    panel_reflectance_by_band.pop(k, None)

            if panel_band_keys:
                status_var.set(f"Detected {len(panel_band_keys)} band(s) in panel folder.")
            else:
                status_var.set("No TIFFs/bands detected in panel folder.")
            return

        # before/after mode: detect bands from both folders and take union
        folder_b = panel_folder_var.get().strip()
        folder_a = panel_folder_after_var.get().strip()

        if not folder_b or not os.path.isdir(folder_b):
            messagebox.showerror("Panels", "Please select a valid before panel folder.")
            panel_band_keys = []
            panel_example_by_band = {}
            return
        if not folder_a or not os.path.isdir(folder_a):
            messagebox.showerror("Panels", "Please select a valid after panel folder.")
            panel_band_keys = []
            panel_example_by_band = {}
            return

        bk_b, ex_b = _detect_bands_from_panel_folder(folder_b)
        bk_a, ex_a = _detect_bands_from_panel_folder(folder_a)

        all_bands = sorted(set(bk_b).union(set(bk_a)))
        example_union = dict(ex_a)
        example_union.update(ex_b)

        panel_band_keys = all_bands
        panel_example_by_band = example_union

        # Clean ROIs / reflectances
        for k in list(panel_roi_by_band_before.keys()):
            if k not in panel_band_keys:
                panel_roi_by_band_before.pop(k, None)
        for k in list(panel_roi_by_band_after.keys()):
            if k not in panel_band_keys:
                panel_roi_by_band_after.pop(k, None)
        for k in list(panel_reflectance_by_band.keys()):
            if k not in panel_band_keys:
                panel_reflectance_by_band.pop(k, None)

        if set(bk_b) != set(bk_a):
            messagebox.showwarning(
                "Panels",
                "Before and after panel folders have different detected bands.\n"
                "We will use the union of bands. Missing bands in after will clamp to before for that band."
            )

        status_var.set(f"Detected {len(panel_band_keys)} band(s) across before/after panel folders.")
        return


    def _select_rois_per_band(which: str = "single"):
        """ROI selection.
        which: 'single' | 'before' | 'after'
        """
        use_ba = bool(panel_before_after_var.get())

        # Decide folder + roi_map
        if use_ba:
            if which == "before":
                folder = panel_folder_var.get().strip()
                roi_map = panel_roi_by_band_before
            elif which == "after":
                folder = panel_folder_after_var.get().strip()
                roi_map = panel_roi_by_band_after
            else:
                # In before/after mode, force explicit choice
                messagebox.showerror("Panels", "In before/after mode, select ROIs for before and after separately.")
                return
        else:
            folder = panel_folder_var.get().strip()
            roi_map = panel_roi_by_band

        if not folder:
            messagebox.showerror("Panels", "Select the panel images folder first.")
            return
        if not os.path.isdir(folder):
            messagebox.showerror("Panels", "Panel folder does not exist.")
            return

        if not panel_band_keys:
            _refresh_panel_bands()
        if not panel_band_keys:
            messagebox.showerror("Panels", "No bands detected in panel folder.")
            return

        # For each band, open a representative panel image and let user pick ROI.
        try:
            import tifffile as tiff
        except Exception as e:
            messagebox.showerror("Panels", f"tifffile is required to read panel TIFFs.\n\n{e!r}")
            return

        # Build example map for this specific folder (so before uses before examples, etc.)
        # We re-detect quickly to avoid cross-folder mismatches.
        _, example_by_band = _detect_bands_from_panel_folder(folder)

        for bk in panel_band_keys:
            path = example_by_band.get(bk)
            if not path or not os.path.exists(path):
                continue
            try:
                arr = tiff.imread(path)
            except Exception as e:
                messagebox.showwarning("Panels", f"Failed to read {os.path.basename(path)}: {e!r}")
                continue

            if arr.ndim > 2:
                arr2d = arr[..., 0]
            else:
                arr2d = arr
            arr2d = np.asarray(arr2d, dtype=np.float32)

            label = which.upper() if use_ba else "SINGLE"
            title = f"Select ROI for {bk} [{label}] ({os.path.basename(path)})"
            roi = _select_roi_matplotlib(arr2d, title=title)
            if roi is None:
                break
            roi_map[bk] = roi

        missing = [bk for bk in panel_band_keys if bk not in roi_map]
        if missing:
            messagebox.showwarning("Panels", f"ROI selection incomplete ({which}). Missing ROIs for: {', '.join(missing)}")
        else:
            messagebox.showinfo("Panels", f"ROIs set for all {len(panel_band_keys)} band(s) ({which}).")


    def _set_reflectance_values():
        """
        Popup allowing:
          - global reflectance (fallback)
          - per-band reflectance overrides (optional)
        """
        if not panel_band_keys:
            _refresh_panel_bands()
        if not panel_band_keys:
            messagebox.showerror("Panels", "No bands detected yet. Select panel folder first.")
            return

        win = tk.Toplevel(frame)
        win.title("Panel reflectance values")
        win.transient(frame.winfo_toplevel())
        win.grab_set()

        ttk.Label(win, text="Global reflectance (fallback for bands without a per-band value):").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        global_entry = ttk.Entry(win, width=10)
        global_entry.grid(row=0, column=1, sticky="w", padx=8, pady=(8, 2))
        global_entry.insert(0, str(panel_reflectance_global_var.get()))

        ttk.Label(win, text="Per-band reflectance (leave blank to use global):").grid(row=1, column=0, sticky="w", padx=8, pady=(6, 2))

        # Scrollable frame
        canvas = tk.Canvas(win, width=420, height=260)
        scroll = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)

        canvas.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=(8, 0), pady=6)
        scroll.grid(row=2, column=2, sticky="ns", padx=(0, 8), pady=6)

        win.grid_rowconfigure(2, weight=1)
        win.grid_columnconfigure(1, weight=1)

        entries = {}
        for i, bk in enumerate(panel_band_keys):
            ttk.Label(inner, text=bk).grid(row=i, column=0, sticky="w", padx=8, pady=2)
            e = ttk.Entry(inner, width=12)
            e.grid(row=i, column=1, sticky="w", padx=8, pady=2)
            if bk in panel_reflectance_by_band:
                e.insert(0, str(panel_reflectance_by_band[bk]))
            entries[bk] = e

        def _apply_and_close():
            # Global
            try:
                g = float(global_entry.get().strip())
            except Exception:
                messagebox.showerror("Reflectance", "Global reflectance must be a number.")
                return
            if not (0.0 < g <= 1.0):
                messagebox.showerror("Reflectance", "Global reflectance must be in (0, 1].")
                return
            panel_reflectance_global_var.set(g)

            # Per-band
            new_map = {}
            for bk, e in entries.items():
                s = e.get().strip()
                if not s:
                    continue
                try:
                    v = float(s)
                except Exception:
                    messagebox.showerror("Reflectance", f"Invalid reflectance for {bk}.")
                    return
                if not (0.0 < v <= 1.0):
                    messagebox.showerror("Reflectance", f"Reflectance for {bk} must be in (0, 1].")
                    return
                new_map[bk] = v

            panel_reflectance_by_band.clear()
            panel_reflectance_by_band.update(new_map)

            win.grab_release()
            win.destroy()

        btn_frame = ttk.Frame(win)
        btn_frame.grid(row=3, column=0, columnspan=3, sticky="e", padx=8, pady=(0, 8))
        ttk.Button(btn_frame, text="Cancel", command=lambda: (win.grab_release(), win.destroy())).grid(row=0, column=0, padx=6)
        ttk.Button(btn_frame, text="Apply", command=_apply_and_close).grid(row=0, column=1)

    # =============================
    # Layout
    # =============================
    row = 0

    ttk.Label(frame, text="Input folder:").grid(row=row, column=0, sticky="w")
    ttk.Entry(frame, textvariable=input_folder_var, width=60).grid(row=row, column=1)
    ttk.Button(frame, text="Browse", command=browse_input).grid(row=row, column=2)
    row += 1

    ttk.Label(frame, text="Output folder:").grid(row=row, column=0, sticky="w")
    ttk.Entry(frame, textvariable=output_folder_var, width=60).grid(row=row, column=1)
    ttk.Button(frame, text="Browse", command=browse_output).grid(row=row, column=2)
    row += 1

    ttk.Label(frame, text="Method:").grid(row=row, column=0, sticky="w")
    method_frame = ttk.Frame(frame)
    method_frame.grid(row=row, column=1, sticky="w")

    ttk.Radiobutton(method_frame, text="Metadata-based", variable=method_var, value="auto").grid(row=0, column=0, sticky="w")
    ttk.Radiobutton(method_frame, text="Metadata + normalization", variable=method_var, value="auto_plus").grid(row=0, column=1, sticky="w")
    ttk.Radiobutton(method_frame, text="Flight log", variable=method_var, value="flight_metadata").grid(row=0, column=2, sticky="w")
    ttk.Radiobutton(method_frame, text="Reflectance panel", variable=method_var, value="calibration_panels").grid(row=0, column=3, sticky="w")
    row += 1

    # Band sensitivity mode (AUTO/AUTO+ only; applied only if metadata exists)
    ttk.Label(frame, text="Band sensitivity:").grid(row=row, column=0, sticky="w")
    bs_combo = ttk.Combobox(frame, textvariable=band_sens_mode_var, width=12, state="readonly")
    bs_combo["values"] = ("Divide", "Multiply", "off")
    bs_combo.grid(row=row, column=1, sticky="w")
    row += 1

    ttk.Checkbutton(frame, text="Recursive", variable=recursive_var).grid(row=row, column=0, sticky="w")
    ttk.Checkbutton(frame, text="Overwrite", variable=overwrite_var).grid(row=row, column=1, sticky="w")
    row += 1

    # Panel controls (only enabled for calibration_panels)
    ttk.Separator(frame, orient="horizontal").grid(row=row, column=0, columnspan=3, sticky="ew", pady=(6, 6))
    row += 1

    ttk.Label(frame, text="Panels captured before and after the flight:").grid(row=row, column=0, sticky="w")
    ba_chk = ttk.Checkbutton(frame, variable=panel_before_after_var, onvalue=True, offvalue=False)
    ba_chk.grid(row=row, column=1, sticky="w")
    row += 1

    ttk.Label(frame, text="Panel images folder (single or before):").grid(row=row, column=0, sticky="w")
    panel_folder_entry = ttk.Entry(frame, textvariable=panel_folder_var, width=60)
    panel_folder_entry.grid(row=row, column=1)
    panel_folder_btn = ttk.Button(frame, text="Browse", command=browse_panel_folder)
    panel_folder_btn.grid(row=row, column=2)
    row += 1

    ttk.Label(frame, text="Panel images folder (after):").grid(row=row, column=0, sticky="w")
    panel_folder_after_entry = ttk.Entry(frame, textvariable=panel_folder_after_var, width=60)
    panel_folder_after_entry.grid(row=row, column=1)
    panel_folder_after_btn = ttk.Button(frame, text="Browse", command=browse_panel_folder_after)
    panel_folder_after_btn.grid(row=row, column=2)
    row += 1

    ttk.Label(frame, text="Panel reflectance (global fallback):").grid(row=row, column=0, sticky="w")
    ttk.Entry(frame, textvariable=panel_reflectance_global_var, width=10).grid(row=row, column=1, sticky="w")
    row += 1

    panel_actions = ttk.Frame(frame)
    panel_actions.grid(row=row, column=1, sticky="w")

    roi_btn_single = ttk.Button(panel_actions, text="Select ROIs (single)…", command=lambda: _select_rois_per_band("single"))
    roi_btn_before = ttk.Button(panel_actions, text="Select ROIs (before)…", command=lambda: _select_rois_per_band("before"))
    roi_btn_after = ttk.Button(panel_actions, text="Select ROIs (after)…", command=lambda: _select_rois_per_band("after"))

    roi_btn_single.grid(row=0, column=0, padx=(0, 8))
    roi_btn_before.grid(row=0, column=0, padx=(0, 8))
    roi_btn_after.grid(row=0, column=1, padx=(0, 8))

    refl_btn = ttk.Button(panel_actions, text="Set reflectance values…", command=_set_reflectance_values)
    refl_btn.grid(row=0, column=2)
    row += 1

    ttk.Label(frame, textvariable=status_var, foreground="gray").grid(row=row, column=0, columnspan=3, sticky="w", pady=(4, 0))
    row += 1

    # Enable/disable panel widgets based on method
    def _update_panel_widgets(*_):
        is_panels = (method_var.get() == "calibration_panels")
        state = "normal" if is_panels else "disabled"

        ba_state = state
        ba_chk.configure(state=ba_state)

        use_ba = bool(panel_before_after_var.get()) and is_panels

        # Entries/buttons
        panel_folder_entry.configure(state=state)
        panel_folder_btn.configure(state=state)

        panel_folder_after_entry.configure(state=("normal" if use_ba else "disabled"))
        panel_folder_after_btn.configure(state=("normal" if use_ba else "disabled"))

        # ROI buttons
        roi_btn_single.configure(state=("normal" if (is_panels and not use_ba) else "disabled"))
        roi_btn_before.configure(state=("normal" if use_ba else "disabled"))
        roi_btn_after.configure(state=("normal" if use_ba else "disabled"))

        refl_btn.configure(state=state)

        # Show/hide ROI buttons cleanly
        if use_ba:
            roi_btn_single.grid_remove()
            roi_btn_before.grid()
            roi_btn_after.grid()
        else:
            roi_btn_before.grid_remove()
            roi_btn_after.grid_remove()
            roi_btn_single.grid()
    method_var.trace_add("write", _update_panel_widgets)
    panel_before_after_var.trace_add("write", _update_panel_widgets)
    _update_panel_widgets()

    # =============================
    # Run logic
    # =============================
    progress = ttk.Progressbar(frame, mode="indeterminate")
    progress.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 4))
    row += 1

    def _run():
        input_folder = input_folder_var.get().strip()
        output_folder = output_folder_var.get().strip() or None
        method = method_var.get().strip()

        if not input_folder or not os.path.isdir(input_folder):
            messagebox.showerror("Radiometric", "Please select a valid input folder.")
            return

        if output_folder and not os.path.isdir(output_folder):
            messagebox.showerror("Radiometric", "Please select a valid output folder.")
            return

        # Validate panel workflow
        panel_folder = None
        panel_folder_after = None
        use_ba = False

        if method == "calibration_panels":
            use_ba = bool(panel_before_after_var.get())

            if use_ba:
                panel_folder = panel_folder_var.get().strip()          # before
                panel_folder_after = panel_folder_after_var.get().strip()  # after

                if not panel_folder or not os.path.isdir(panel_folder):
                    messagebox.showerror("Panels", "Before/After panels: please select a valid before panel folder.")
                    return
                if not panel_folder_after or not os.path.isdir(panel_folder_after):
                    messagebox.showerror("Panels", "Before/After panels: please select a valid after panel folder.")
                    return

                if not panel_band_keys:
                    _refresh_panel_bands()

                # Require ROIs for BOTH sets (only for detected bands)
                missing_before = [bk for bk in panel_band_keys if bk not in panel_roi_by_band_before]
                missing_after = [bk for bk in panel_band_keys if bk not in panel_roi_by_band_after]
                if missing_before or missing_after:
                    msg = []
                    if missing_before:
                        msg.append("Missing before ROIs for:\n" + "\n".join(missing_before))
                    if missing_after:
                        msg.append("Missing after ROIs for:\n" + "\n".join(missing_after))
                    messagebox.showerror("Panels", "\n\n".join(msg))
                    return

            else:
                panel_folder = panel_folder_var.get().strip()
                if not panel_folder or not os.path.isdir(panel_folder):
                    messagebox.showerror("Panels", "Calibration panels requires selecting a valid panel images folder.")
                    return

                if not panel_band_keys:
                    _refresh_panel_bands()

                missing_rois = [bk for bk in panel_band_keys if bk not in panel_roi_by_band]
                if missing_rois:
                    messagebox.showerror(
                        "Panels",
                        "You must select ONE ROI per band before running.\n\nMissing ROIs for:\n" + "\n".join(missing_rois)
                    )
                    return

        cfg = RadiometricConfig(
            input_folder=input_folder,
            output_folder=output_folder,
            method=method,
            recursive=bool(recursive_var.get()),
            overwrite=bool(overwrite_var.get()),
            panel_reflectance=float(panel_reflectance_global_var.get()),
            panel_folder=panel_folder,
            panel_roi_by_band=dict(panel_roi_by_band) if panel_roi_by_band else None,
            panel_roi_by_band_before=dict(panel_roi_by_band_before) if panel_roi_by_band_before else None,
            panel_roi_by_band_after=dict(panel_roi_by_band_after) if panel_roi_by_band_after else None,
            panel_before_after=bool(use_ba),
            panel_folder_before=panel_folder if use_ba else None,
            panel_folder_after=panel_folder_after if use_ba else None,
            panel_reflectance_by_band=dict(panel_reflectance_by_band) if panel_reflectance_by_band else None,
            band_sensitivity_mode=str(band_sens_mode_var.get()).strip().lower(),
        )

        def worker():
            try:
                apply_radiometric_corrections(cfg)
                messagebox.showinfo("Radiometric", "Radiometric correction finished.")
            except Exception as e:
                messagebox.showerror("Radiometric", f"Radiometric correction failed:\n\n{e!r}")
            finally:
                progress.stop()
                run_btn.configure(state="normal")
                status_var.set("Ready.")

        progress.start(10)
        run_btn.configure(state="disabled")
        status_var.set("Running…")
        threading.Thread(target=worker, daemon=True).start()

    run_btn = ttk.Button(frame, text="Run radiometric correction", command=_run)
    run_btn.grid(row=row, column=0, sticky="w", pady=(4, 8))

    return frame


def criar_aba_radiometric(parent):
    frame = create_radiometric_tab(parent)
    frame.pack(fill="both", expand=True)
    return frame
