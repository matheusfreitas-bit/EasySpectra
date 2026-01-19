# analise_espectral.py
#
# Spectral analysis tools for EasySpectra.
# NOTE: Function names and external API are preserved.
# Only UI texts, comments and documentation were adapted
# for an English, commercially-oriented version.

import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog, Toplevel, Listbox, Scrollbar, Button
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, EllipseSelector, PolygonSelector
from matplotlib.path import Path
import os
import json

from .funcoes_importacao import get_cube, get_wavelengths
from .normalizacao import normalizar_cubo


# Global variables (kept as simple state holders for the GUI workflow)
cube = None
wavelengths = None
rgb_norm = None
zoomed_cube = None
zoomed_rgb = None
cube_area_selecionada = None  # subcube (rectangular region)
area_espectros_2d = None      # spectra (Npix, B) extracted from ellipse/polygon (or processed rectangle)
metodos_escolhidos = []
bandas_rgb = []
bandas_rgb_definidas = False
selector_zoom = None

# Keep selectors alive (avoid garbage collection issues)
_seletores_ativos = []


def _limpar_seletores():
    """Disconnect and clear active selectors to avoid conflicts and GC issues."""
    global _seletores_ativos
    try:
        for s in _seletores_ativos:
            try:
                s.disconnect_events()
            except Exception:
                pass
    finally:
        _seletores_ativos = []



from .dataset_manager import save_dataset_rows

def _dataset_ask_export_type():
    win = Toplevel()
    win.title("Dataset export type")
    win.geometry("360x220")
    win.grab_set()

    tk.Label(win, text="What do you want to export?").pack(pady=10)
    choice = tk.StringVar(value="mean")

    tk.Radiobutton(win, text="Only mean spectrum", variable=choice, value="mean").pack(anchor="w", padx=18)
    tk.Radiobutton(win, text="Individual pixels", variable=choice, value="individual").pack(anchor="w", padx=18)

    result = {"val": None}

    def _ok():
        result["val"] = choice.get()
        win.destroy()

    def _cancel():
        result["val"] = None
        win.destroy()

    btns = tk.Frame(win)
    btns.pack(pady=14)
    tk.Button(btns, text="Continue", command=_ok).pack(side="left", padx=8)
    tk.Button(btns, text="Cancel", command=_cancel).pack(side="left", padx=8)

    win.wait_window()
    return result["val"]


def _dataset_ask_pixel_mode():
    win = Toplevel()
    win.title("Pixel export mode")
    win.geometry("380x260")
    win.grab_set()

    tk.Label(win, text="How do you want to export the pixels?").pack(pady=10)
    mode = tk.StringVar(value="all")

    tk.Radiobutton(win, text="All pixels", variable=mode, value="all").pack(anchor="w", padx=18)
    tk.Radiobutton(win, text="Random sampling", variable=mode, value="sample").pack(anchor="w", padx=18)

    n_var = tk.StringVar(value="1000")
    frm = tk.Frame(win)
    frm.pack(anchor="w", padx=18, pady=10)
    tk.Label(frm, text="If sampling, number of pixels:").pack(side="left")
    tk.Entry(frm, textvariable=n_var, width=8).pack(side="left", padx=8)

    result = {"mode": None, "n": None}

    def _ok():
        m = mode.get()
        if m == "sample":
            try:
                n = int(n_var.get())
                if n <= 0:
                    raise ValueError
            except Exception:
                messagebox.showerror("Error", "Please enter a valid positive integer.")
                return
            result["mode"] = "sample"
            result["n"] = n
        else:
            result["mode"] = "all"
            result["n"] = None
        win.destroy()

    def _cancel():
        win.destroy()

    btns = tk.Frame(win)
    btns.pack(pady=14)
    tk.Button(btns, text="Continue", command=_ok).pack(side="left", padx=8)
    tk.Button(btns, text="Cancel", command=_cancel).pack(side="left", padx=8)

    win.wait_window()
    return result["mode"], result["n"]


def _dataset_build_header():
    cols = [f"{float(wl):.1f}nm" for wl in wavelengths]
    cols.append("label")
    return cols


def _dataset_rows_from_spectra(spectra, label):
    rows = []
    for i in range(spectra.shape[0]):
        row = [float(v) for v in spectra[i, :]]
        row.append(label)
        rows.append(row)
    return rows


def _dataset_export(mode: str):
    global area_espectros_2d

    if area_espectros_2d is None:
        messagebox.showerror("Error", "No area spectra available. Perform area analysis first.")
        return

    label = simpledialog.askstring("Label", "Enter class/label name:")
    if not label:
        return
    label = label.strip()

    export_type = _dataset_ask_export_type()
    if export_type is None:
        return

    spectra = area_espectros_2d

    warning_msg = None

    if export_type == "mean":
        chosen = spectra.mean(axis=0).reshape(1, -1)
        saved_n = 1
    else:
        pixel_mode, n_req = _dataset_ask_pixel_mode()
        if pixel_mode is None:
            return

        n_pix = spectra.shape[0]

        if pixel_mode == "all":
            chosen = spectra
            saved_n = n_pix
        else:
            if n_pix <= n_req:
                chosen = spectra
                saved_n = n_pix
                warning_msg = f"ROI has only {n_pix} pixels. All were saved."
            else:
                idx = np.random.choice(n_pix, size=n_req, replace=False)
                chosen = spectra[idx, :]
                saved_n = chosen.shape[0]

    if mode == "create":
        csv_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    else:
        csv_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

    if not csv_path:
        return

    header = _dataset_build_header()
    rows = _dataset_rows_from_spectra(chosen, label)

    try:
        save_dataset_rows(csv_path, header, rows, mode)
    except Exception as e:
        messagebox.showerror("Dataset error", f"{type(e).__name__}: {e}")
        return

    if warning_msg:
        messagebox.showwarning("Sampling notice", warning_msg)

    messagebox.showinfo("Dataset saved", f"Saved {saved_n} observation(s).")


def criar_dataset_espectral_gui():
    _dataset_export("create")


def adicionar_dataset_espectral_gui():
    _dataset_export("append")

# --- SELECT RGB BANDS ---
def selecionar_bandas_rgb(force=False):
    """
    Open a dialog to select 1–3 bands to be used as RGB (or single-band) visualization.

    Parameters
    ----------
    force : bool, optional
        If True, forces the dialog even if RGB bands were already defined.
    """
    global bandas_rgb, bandas_rgb_definidas

    if bandas_rgb_definidas and not force:
        return

    if cube is None or wavelengths is None:
        messagebox.showerror("Error", "No spectral cube is currently loaded.")
        return

    janela_bandas = Toplevel()
    janela_bandas.title("Select bands for visualization")
    janela_bandas.geometry("320x400")
    janela_bandas.grab_set()

    frame_scroll = tk.Frame(janela_bandas)
    frame_scroll.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    scrollbar = Scrollbar(frame_scroll)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    lista = Listbox(
        frame_scroll,
        selectmode=tk.MULTIPLE,
        yscrollcommand=scrollbar.set,
        height=20,
        width=30,
    )
    for i, wl in enumerate(wavelengths):
        lista.insert(tk.END, f"{i}: {wl:.1f} nm")
    lista.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=lista.yview)

    def confirmar():
        indices = lista.curselection()
        if len(indices) not in [1, 2, 3]:
            messagebox.showerror("Error", "Please select 1, 2, or 3 bands.")
            return
        bandas_rgb.clear()
        bandas_rgb.extend(indices)
        bandas_rgb_definidas = True
        janela_bandas.destroy()
        visualizar_rgb_principal()

    botao = Button(janela_bandas, text="Confirm", command=confirmar)
    botao.pack(pady=10)


# --- MAIN RGB VISUALIZATION ---
def visualizar_rgb_principal():
    """
    Display the main RGB (or single/dual band) image for the current cube
    based on the selected bands in `bandas_rgb`.
    """
    global rgb_norm

    imagem = cube[:, :, bandas_rgb].astype(float)

    if len(bandas_rgb) == 1:
        # Single-band grayscale
        imagem = imagem[:, :, 0]
    elif len(bandas_rgb) == 2:
        # Use two bands and set the third as zeros
        imagem = np.stack(
            [imagem[:, :, 0], imagem[:, :, 1], np.zeros_like(imagem[:, :, 0])],
            axis=-1,
        )

    # Simple min-max scaling to [0, 1]
    rgb_norm = (imagem - imagem.min()) / (imagem.max() - imagem.min() + 1e-12)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb_norm)
    ax.set_title(
        f"Bands: {[round(wavelengths[i], 1) for i in bandas_rgb]} (R-G-B mapping)"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# --- IMPORT CUBE FOR ANALYSIS ---
def importar_para_analise():
    """
    Import a spectral cube (.npy) and its associated metadata (.json),
    or interactively define the wavelength range if the JSON is missing.
    """
    global cube, wavelengths, bandas_rgb, bandas_rgb_definidas

    caminho = filedialog.askopenfilename(
        title="Select spectral cube (.npy)",
        filetypes=[("NumPy array files", "*.npy")],
    )
    if not caminho:
        return

    cube = np.load(caminho)
    json_path = caminho.replace(".npy", ".json")

    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                metadata = json.load(f)
                wavelengths = np.array(metadata["wavelengths"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load JSON metadata:\n{e}")
            return
    else:
        comprimento_min = simpledialog.askinteger(
            "Minimum wavelength",
            "Enter the minimum wavelength (nm):",
            initialvalue=400,
        )
        comprimento_max = simpledialog.askinteger(
            "Maximum wavelength",
            "Enter the maximum wavelength (nm):",
            initialvalue=1000,
        )
        if comprimento_min is None or comprimento_max is None:
            messagebox.showwarning(
                "Cancelled",
                "Import cancelled – wavelength range was not defined.",
            )
            return
        bandas = cube.shape[2]
        wavelengths = np.linspace(comprimento_min, comprimento_max, bandas)

    messagebox.showinfo(
        "Success",
        f"Spectral cube with {cube.shape[2]} bands successfully loaded!",
    )
    bandas_rgb_definidas = False
    selecionar_bandas_rgb(force=True)


# --- PIXEL-BASED ANALYSIS ---
def on_click_zoom(event):
    """
    Handle mouse click on the zoomed RGB image to display
    the spectrum of the selected pixel.
    """
    if event.inaxes and zoomed_cube is not None:
        x = int(event.xdata)
        y = int(event.ydata)
        if (
            y < 0
            or x < 0
            or y >= zoomed_cube.shape[0]
            or x >= zoomed_cube.shape[1]
        ):
            return

        espectro = zoomed_cube[y, x, :]

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(wavelengths, espectro)
        ax2.set_title(
            f"Pixel spectrum at ({y}, {x}) - Methods: {metodos_escolhidos}"
        )
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Intensity")
        ax2.grid(True)
        plt.tight_layout()
        plt.show()


def on_select_zoom_pixel(ecanto, fcanto):
    """
    RectangleSelector callback to define a zoom region for pixel-based analysis.
    """
    global zoomed_cube, zoomed_rgb

    if ecanto.xdata is None or fcanto.xdata is None:
        return

    x1, y1 = int(ecanto.xdata), int(ecanto.ydata)
    x2, y2 = int(fcanto.xdata), int(fcanto.ydata)

    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    cubo_zoom = cube[ymin:ymax, xmin:xmax, :]
    zoomed_cube = normalizar_cubo(cubo_zoom, metodo=metodos_escolhidos)

    imagem_zoom = zoomed_cube[:, :, bandas_rgb].astype(float)
    zoomed_rgb = (imagem_zoom - imagem_zoom.min()) / (
        imagem_zoom.max() - imagem_zoom.min() + 1e-12
    )

    fig_zoom, ax_zoom = plt.subplots(figsize=(8, 6))
    ax_zoom.imshow(zoomed_rgb)
    ax_zoom.set_title("Zoomed region – click to inspect pixel spectra")
    ax_zoom.axis("off")
    fig_zoom.canvas.mpl_connect("button_press_event", on_click_zoom)
    plt.show()


def analise_por_pixel():
    """
    High-level workflow for pixel-based spectral analysis:
    1) Display RGB image for the full cube;
    2) User selects a zoom rectangle;
    3) Within the zoom, user clicks on pixels to inspect spectra.
    """
    global metodos_escolhidos, selector_zoom

    if not bandas_rgb:
        selecionar_bandas_rgb(force=True)

    metodos_escolhidos = []

    imagem = cube[:, :, bandas_rgb].astype(float)
    rgb_img = (imagem - imagem.min()) / (
        imagem.max() - imagem.min() + 1e-12
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb_img)
    ax.set_title("1️⃣ Select a region to zoom in")
    ax.axis("off")

    _limpar_seletores()
    selector_zoom = RectangleSelector(
        ax,
        on_select_zoom_pixel,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
        props=dict(facecolor="none", edgecolor="yellow", linewidth=1.5),
    )
    _seletores_ativos.append(selector_zoom)
    plt.show()


# --- AREA-BASED ANALYSIS ---
def _escolher_modo_selecao_area():
    """
    Open a small dialog to select the area selection mode
    (rectangle, ellipse or polygon).
    """
    win = Toplevel()
    win.title("Area selection tool")
    win.geometry("320x200")
    win.grab_set()

    tk.Label(win, text="Choose the area selection mode:").pack(pady=8)

    modo = tk.StringVar(value="ret")
    options = [
        ("Rectangle", "ret"),
        ("Circle/Ellipse", "circ"),
        ("Polygon (click vertices, double-click to finish)", "pol"),
    ]
    for txt, val in options:
        tk.Radiobutton(
            win,
            text=txt,
            variable=modo,
            value=val,
            anchor="w",
            justify="left",
            wraplength=300,
        ).pack(fill="x", padx=14)

    Button(win, text="Continue", command=win.destroy).pack(pady=10)
    win.wait_window()
    return modo.get()


def _plot_area_espectros(espectros, titulo="Spectra from selected area"):
    """
    Plot all individual spectra from the selected area,
    together with their mean spectrum.
    """
    media = espectros.mean(axis=0)
    fig, ax = plt.subplots(figsize=(10, 5))

    for espectro in espectros:
        ax.plot(
            wavelengths,
            espectro,
            color="lightgray",
            linewidth=0.6,
            alpha=0.6,
        )

    ax.plot(
        wavelengths,
        media,
        color="black",
        linewidth=2,
        label="Mean spectrum",
    )

    ax.set_title(titulo)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def on_select_area_retangulo(ecanto, fcanto):
    """
    RectangleSelector callback (within the zoomed image) to define an area
    and extract spectra for all pixels in that area.
    """
    global cube_area_selecionada, area_espectros_2d

    if ecanto.xdata is None or fcanto.xdata is None:
        print("[EasySpectra] Warning: invalid selection – please click inside the image.")
        return

    x1, y1 = int(ecanto.xdata), int(ecanto.ydata)
    x2, y2 = int(fcanto.xdata), int(fcanto.ydata)

    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    cube_area_selecionada = zoomed_cube[ymin:ymax, xmin:xmax, :]

    if cube_area_selecionada.size == 0:
        return

    h, w, b = cube_area_selecionada.shape
    espectros = cube_area_selecionada.reshape(h * w, b)
    area_espectros_2d = espectros

    _plot_area_espectros(
        espectros,
        "Spectra from selected area (rectangle)",
    )


def _ativar_seletor_elipse(ax_zoom_img):
    """
    Enable elliptical selection on the zoomed image.
    The resulting spectra (Npix, B) are stored in `area_espectros_2d`.
    """

    def _on_ellipse(ec, fc):
        global area_espectros_2d

        if ec.xdata is None or fc.xdata is None:
            return

        x1, y1 = ec.xdata, ec.ydata
        x2, y2 = fc.xdata, fc.ydata

        xmn, xmx = sorted([x1, x2])
        ymn, ymx = sorted([y1, y2])

        cx = (xmn + xmx) / 2.0
        cy = (ymn + ymx) / 2.0
        rx = abs(xmx - xmn) / 2.0
        ry = abs(ymx - ymn) / 2.0

        if rx < 1e-6 or ry < 1e-6:
            return

        h, w = zoomed_cube.shape[:2]
        Y, X = np.ogrid[:h, :w]
        mask = (((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2) <= 1.0

        sel = zoomed_cube[mask, :]  # (Npix, B)
        if sel.size == 0:
            return

        area_espectros_2d = sel
        _plot_area_espectros(
            sel,
            "Spectra from selected area (ellipse)",
        )

    sel = EllipseSelector(
        ax_zoom_img,
        _on_ellipse,
        useblit=True,
        button=[1],
        interactive=True,
        props=dict(facecolor="none", edgecolor="yellow", linewidth=1.5),
    )
    _seletores_ativos.append(sel)


def _ativar_seletor_poligono(ax_zoom_img):
    """
    Enable polygon selection on the zoomed image.

    Click to add vertices and double-click to close the polygon.
    Spectra are extracted for all pixels inside the polygon.
    """

    def _on_polygon(verts):
        global area_espectros_2d

        if not verts or len(verts) < 3:
            return

        h, w = zoomed_cube.shape[:2]
        p = Path(verts)
        Y, X = np.mgrid[:h, :w]
        pts = np.vstack((X.ravel(), Y.ravel())).T
        mask = p.contains_points(pts).reshape(h, w)

        sel = zoomed_cube[mask, :]  # (Npix, B)
        if sel.size == 0:
            return

        area_espectros_2d = sel
        _plot_area_espectros(
            sel,
            "Spectra from selected area (polygon)",
        )

    # PolygonSelector relies on Line2D → use `color` instead of face/edgecolor
    sel = PolygonSelector(
        ax_zoom_img,
        _on_polygon,
        useblit=True,
        props=dict(color="yellow", linewidth=1.5),
    )
    _seletores_ativos.append(sel)


def on_select_zoom_area(ecanto, fcanto):
    """
    RectangleSelector callback to define the zoom region for area-based analysis.
    After zooming, the user chooses the selection tool (rectangle, ellipse, or polygon)
    to extract spectra from a specific area within the zoomed region.
    """
    global zoomed_cube, zoomed_rgb, selector_zoom, area_espectros_2d, cube_area_selecionada

    if ecanto.xdata is None or fcanto.xdata is None:
        return

    x1, y1 = int(ecanto.xdata), int(ecanto.ydata)
    x2, y2 = int(fcanto.xdata), int(fcanto.ydata)

    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    cubo_zoom = cube[ymin:ymax, xmin:xmax, :]
    if cubo_zoom.size == 0:
        return

    zoomed_cube = normalizar_cubo(cubo_zoom, metodo=metodos_escolhidos)

    imagem_zoom = zoomed_cube[:, :, bandas_rgb].astype(float)
    zoomed_rgb = (imagem_zoom - imagem_zoom.min()) / (
        imagem_zoom.max() - imagem_zoom.min() + 1e-12
    )

    fig_zoom, ax_zoom = plt.subplots(figsize=(10, 8))
    ax_zoom.imshow(zoomed_rgb)
    ax_zoom.set_title("2️⃣ Select an area to extract spectra")
    ax_zoom.axis("off")

    # Select area selection mode after zoom
    modo = _escolher_modo_selecao_area()
    area_espectros_2d = None
    cube_area_selecionada = None

    _limpar_seletores()
    if modo == "ret":
        sel = RectangleSelector(
            ax_zoom,
            on_select_area_retangulo,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
            props=dict(facecolor="none", edgecolor="yellow", linewidth=1.5),
        )
        _seletores_ativos.append(sel)
    elif modo == "circ":
        _ativar_seletor_elipse(ax_zoom)
    else:  # "pol"
        _ativar_seletor_poligono(ax_zoom)

    plt.show()


def analise_por_area():
    """
    High-level workflow for area-based spectral analysis:
    1) Display RGB image for the full cube;
    2) User selects a zoom rectangle;
    3) Within the zoom, user chooses an area selection tool;
    4) Spectra are extracted for the selected area.
    """
    global metodos_escolhidos, selector_zoom

    if not bandas_rgb:
        selecionar_bandas_rgb(force=True)

    metodos_escolhidos = []

    imagem = cube[:, :, bandas_rgb].astype(float)
    rgb_img = (imagem - imagem.min()) / (
        imagem.max() - imagem.min() + 1e-12
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb_img)
    ax.set_title("1️⃣ Select a region to zoom in")
    ax.axis("off")

    _limpar_seletores()
    selector_zoom = RectangleSelector(
        ax,
        on_select_zoom_area,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
        props=dict(facecolor="none", edgecolor="yellow", linewidth=1.5),
    )
    _seletores_ativos.append(selector_zoom)
    plt.show()


# --- EXPORT AREA SPECTRA TO CSV ---
def exportar_espectros_area_csv():
    """
    Export all spectra from the selected area (and the mean spectrum)
    to a CSV file. Each row corresponds to one pixel spectrum.
    """
    global area_espectros_2d, cube_area_selecionada

    if area_espectros_2d is None and cube_area_selecionada is None:
        messagebox.showerror(
            "Error",
            "No area has been selected yet. Please perform area analysis first.",
        )
        return

    try:
        if area_espectros_2d is not None:
            espectros = area_espectros_2d  # (Npix, B)
        else:
            h, w, b = cube_area_selecionada.shape
            espectros = cube_area_selecionada.reshape(h * w, b)

        media = espectros.mean(axis=0)

        caminho = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            title="Save spectra as CSV",
        )
        if not caminho:
            return

        with open(caminho, "w") as f:
            header = ",".join([f"{wl:.1f}nm" for wl in wavelengths])
            f.write(f"Type,{header}\n")
            for esp in espectros:
                linha = ",".join(map(str, esp))
                f.write(f"Pixel,{linha}\n")
            linha_media = ",".join(map(str, media))
            f.write(f"Mean,{linha_media}\n")

        messagebox.showinfo(
            "Success",
            "Area spectra were successfully exported to CSV.",
        )
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Failed to export spectra to CSV:\n{e}",
        )


# --- MAIN SPECTRAL ANALYSIS TAB ---
def criar_aba_analise_espectral(aba):
    """
    Create the main 'Spectral Analysis' tab inside the provided parent widget.
    """
    frame = ttk.Frame(aba)
    frame.pack(anchor="w", padx=20, pady=20)

    tk.Button(
        frame,
        text="📂 Import cube (.npy) for analysis",
        command=importar_para_analise,
        font=("Arial", 12),
    ).pack(anchor="w", pady=5)

    tk.Button(
        frame,
        text="🎨 Select and vizualize bands",
        command=lambda: selecionar_bandas_rgb(force=True),
        font=("Arial", 12),
    ).pack(anchor="w", pady=5)

    tk.Button(
        frame,
        text="🔍 Pixel-based spectral analysis",
        command=analise_por_pixel,
        font=("Arial", 12),
    ).pack(anchor="w", pady=5)

    tk.Button(
        frame,
        text="🗂️ Area-based spectral analysis",
        command=analise_por_area,
        font=("Arial", 12),
    ).pack(anchor="w", pady=5)

    tk.Button(
        frame,
        text="💾 Export spectral data to CSV",
        command=exportar_espectros_area_csv,
        font=("Arial", 12),
    ).pack(anchor="w", pady=5)





    tk.Button(
        frame,
        text="🧱 Create dataset (labeled CSV)",
        command=criar_dataset_espectral_gui,
        font=("Arial", 12),
    ).pack(anchor="w", pady=5)

    tk.Button(
        frame,
        text="➕ Append to existing dataset",
        command=adicionar_dataset_espectral_gui,
        font=("Arial", 12),
    ).pack(anchor="w", pady=5)