# indices_vegetacao.py
#
# Vegetation indices tools and GUI tab for EasySpectra.
#
# Workflow:
# 1) Import spectral cube (.npy + .json with "wavelengths").
# 2) Compute standard vegetation indices (NDVI, GNDVI, NDRE).
# 3) Create custom indices based on an arbitrary mathematical expression
#    using any number of bands (b0, b1, b2, ...).
# 4) Visualize selected index as image.
# 5) Zoom on the index and select an area using:
#    - Rectangle
#    - Ellipse (circle)
#    - Polygon (free shape)
# 6) After area selection, ask user if they want to export CSV.
# 7) Export selected area values to CSV, if requested.

import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Toplevel, Listbox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, EllipseSelector, PolygonSelector
from matplotlib.path import Path
import os
import json


# ---------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------

cube_vi = None              # spectral cube for vegetation indices (H, W, B)
wavelengths_vi = None       # wavelengths (B,)
indices_dict = {}           # name -> 2D array (H, W)

indice_nome_atual = None    # currently selected index name
indice_imagem_atual = None  # full index image (H, W)
indice_zoomada = None       # zoomed index subimage (h, w)
indice_area_valores = None  # 1D array (Npix,) of values from selected area

_seletores_ativos_vi = []   # keep selectors alive (avoid garbage collection)


def _limpar_seletores_vi():
    """Disconnect and clear active selectors to avoid conflicts and GC issues."""
    global _seletores_ativos_vi
    try:
        for s in _seletores_ativos_vi:
            try:
                s.disconnect_events()
            except Exception:
                pass
    finally:
        _seletores_ativos_vi = []


# ---------------------------------------------------------------------
# CORE HELPERS
# ---------------------------------------------------------------------

def _garantir_cubo_carregado():
    """
    Ensure that a spectral cube is loaded.
    If not, ask the user to import a cube.

    Returns
    -------
    bool
        True if a cube is available after this call, False otherwise.
    """
    global cube_vi, wavelengths_vi

    if cube_vi is not None and wavelengths_vi is not None:
        return True

    # Try to import interactively
    importar_cubo_para_indices()
    if cube_vi is None or wavelengths_vi is None:
        messagebox.showerror(
            "Error",
            "No spectral cube is available. Please import a cube first.",
        )
        return False
    return True


def _closest_band_idx(target_nm: float) -> int:
    """
    Return the index of the band whose wavelength is closest to target_nm.
    """
    if wavelengths_vi is None or len(wavelengths_vi) == 0:
        raise ValueError("Wavelengths are not defined.")
    diffs = np.abs(wavelengths_vi - target_nm)
    return int(np.argmin(diffs))


# ---------------------------------------------------------------------
# IMPORT CUBE (.NPY + .JSON)
# ---------------------------------------------------------------------

def importar_cubo_para_indices():
    """
    Import a spectral cube (.npy) and its associated metadata (.json)
    with a 'wavelengths' field or interactively define the wavelength range.
    """
    global cube_vi, wavelengths_vi, indices_dict

    caminho = filedialog.askopenfilename(
        title="Select spectral cube (.npy)",
        filetypes=[("NumPy array files", "*.npy")],
    )
    if not caminho:
        return

    try:
        cube_vi = np.load(caminho)
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Failed to load .npy cube:\n{type(e).__name__}: {e}",
        )
        cube_vi = None
        wavelengths_vi = None
        return

    json_path = caminho.replace(".npy", ".json")
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                metadata = json.load(f)
                wavelengths_vi = np.array(metadata["wavelengths"])
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to load JSON metadata:\n{type(e).__name__}: {e}",
            )
            cube_vi = None
            wavelengths_vi = None
            return
    else:
        from tkinter import simpledialog

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
            cube_vi = None
            wavelengths_vi = None
            return
        bandas = cube_vi.shape[2]
        wavelengths_vi = np.linspace(comprimento_min, comprimento_max, bandas)

    # Reset indices
    indices_dict.clear()
    messagebox.showinfo(
        "Success",
        f"Spectral cube for vegetation indices loaded with {cube_vi.shape[2]} bands.",
    )


# ---------------------------------------------------------------------
# VEGETATION INDICES ENGINE
# ---------------------------------------------------------------------

def calcular_indices_padrao():
    """
    Compute a set of standard vegetation indices for the current cube:
    - NDVI (NIR and Red)
    - GNDVI (NIR and Green)
    - NDRE (NIR and Red Edge)

    Results are stored in the global dictionary indices_dict
    as 2D arrays with shape (H, W).
    """
    global indices_dict

    if not _garantir_cubo_carregado():
        return

    if cube_vi.ndim != 3:
        messagebox.showerror(
            "Error",
            "The spectral cube has an invalid shape. Expected a 3D array (H, W, B).",
        )
        return

    data = cube_vi.astype(float)
    h, w, b = data.shape
    if b < 2:
        messagebox.showerror(
            "Error",
            "The cube must have at least 2 bands to compute vegetation indices.",
        )
        return

    EPS = 1e-12

    try:
        # Approximate wavelengths (nm) for typical bands
        red_idx = _closest_band_idx(660.0)       # Red
        nir_idx = _closest_band_idx(800.0)       # NIR
        green_idx = _closest_band_idx(560.0)     # Green
        red_edge_idx = _closest_band_idx(705.0)  # Red edge

        R = data[:, :, red_idx]
        NIR = data[:, :, nir_idx]
        G = data[:, :, green_idx]
        RE = data[:, :, red_edge_idx]

        ndvi = (NIR - R) / (NIR + R + EPS)
        gndvi = (NIR - G) / (NIR + G + EPS)
        ndre = (NIR - RE) / (NIR + RE + EPS)

        indices_dict["NDVI"] = ndvi
        indices_dict["GNDVI"] = gndvi
        indices_dict["NDRE"] = ndre

        messagebox.showinfo(
            "Vegetation indices",
            "Standard vegetation indices were computed:\n- NDVI\n- GNDVI\n- NDRE",
        )
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Failed to compute vegetation indices:\n{type(e).__name__}: {e}",
        )


def criar_indice_personalizado(nome: str, expressao: str):
    """
    Create a custom index based on an arbitrary mathematical expression.

    Parameters
    ----------
    nome : str
        Name of the new index (key in indices_dict).
    expressao : str
        Mathematical expression that should evaluate to a 2D array (H, W).
        Bands are referenced as:
            b0, b1, b2, ..., b(N-1)
        where bi is the 2D array for band i.

    Examples
    --------
    - (b8 - b3) / (b8 + b3 + 1e-12)
    - np.log(b10 + 1) - np.log(b2 + 1)
    - (b5 - b4) / (b5 + b4 + 1e-12)
    """
    global indices_dict

    if not _garantir_cubo_carregado():
        return

    data = cube_vi.astype(float)
    h, w, b = data.shape

    # Build a safe evaluation environment
    env = {
        "np": np,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "where": np.where,
    }

    # Add band variables: b0, b1, b2, ...
    for i in range(b):
        env[f"b{i}"] = data[:, :, i]

    try:
        arr = eval(expressao, {"__builtins__": {}}, env)
    except Exception as e:
        messagebox.showerror(
            "Custom index",
            f"Error evaluating expression:\n{type(e).__name__}: {e}",
        )
        return

    arr = np.array(arr, dtype=float)
    if arr.shape != (h, w):
        messagebox.showerror(
            "Custom index",
            f"Expression result has shape {arr.shape}, but expected {(h, w)}.",
        )
        return

    indices_dict[nome] = arr
    messagebox.showinfo(
        "Custom index",
        f"Custom index '{nome}' was successfully created.",
    )


def visualizar_indice(nome_indice: str):
    """
    Visualize a vegetation index (2D array) as an image.

    Parameters
    ----------
    nome_indice : str
        Name of the index in indices_dict.
    """
    global indice_nome_atual, indice_imagem_atual

    if nome_indice not in indices_dict:
        messagebox.showerror(
            "Error",
            f"The index '{nome_indice}' is not available.",
        )
        return

    img = indices_dict[nome_indice]
    indice_nome_atual = nome_indice
    indice_imagem_atual = img

    if img.ndim != 2:
        messagebox.showerror(
            "Error",
            "The selected index does not have 2D shape (H, W).",
        )
        return

    vmin = np.nanpercentile(img, 2)
    vmax = np.nanpercentile(img, 98)
    if vmax <= vmin:
        vmin = np.nanmin(img)
        vmax = np.nanmax(img)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(img, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_title(f"Vegetation index: {nome_indice}")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# EXPORT AREA VALUES TO CSV
# ---------------------------------------------------------------------

def exportar_indice_area_csv():
    """
    Export the values of the currently selected area (over an index image)
    to a CSV file. One value per row.

    Columns:
    - IndexName
    - Value
    """
    global indice_area_valores, indice_nome_atual

    if indice_area_valores is None or indice_nome_atual is None:
        messagebox.showerror(
            "Error",
            "No area has been selected on an index yet. "
            "Please select an area first.",
        )
        return

    caminho = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save index area data as CSV",
    )
    if not caminho:
        return

    try:
        with open(caminho, "w") as f:
            f.write("IndexName,Value\n")
            for v in indice_area_valores:
                f.write(f"{indice_nome_atual},{v}\n")

        messagebox.showinfo(
            "Success",
            "Vegetation index values were successfully exported to CSV.",
        )
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Failed to export vegetation index data to CSV:\n{e}",
        )


# ---------------------------------------------------------------------
# AREA SELECTION TOOLS (RECTANGLE / ELLIPSE / POLYGON)
# ---------------------------------------------------------------------

def _escolher_modo_selecao_indice():
    """
    Open a small dialog to select the area selection mode
    (rectangle, ellipse or polygon) for index maps.
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

    tk.Button(win, text="Continue", command=win.destroy).pack(pady=10)
    win.wait_window()
    return modo.get()


def on_select_area_indice_retangulo(ecanto, fcanto):
    """
    RectangleSelector callback (within the zoomed index image) to define an area
    and extract values for all pixels in that area.
    After selecting, ask whether to save CSV and close the zoom window.
    """
    global indice_zoomada, indice_area_valores

    if ecanto.xdata is None or fcanto.xdata is None:
        print("[EasySpectra] Warning: invalid selection – please click inside the image.")
        return

    x1, y1 = int(ecanto.xdata), int(ecanto.ydata)
    x2, y2 = int(fcanto.xdata), int(fcanto.ydata)

    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    if indice_zoomada is None:
        return

    sub = indice_zoomada[ymin:ymax, xmin:xmax]
    if sub.size == 0:
        return

    indice_area_valores = sub.reshape(-1)
    n = indice_area_valores.size

    fig = ecanto.inaxes.figure if ecanto.inaxes is not None else None

    resp = messagebox.askyesno(
        "Save index area data?",
        f"{n} pixels were selected on index '{indice_nome_atual}'.\n\n"
        "Do you want to export these values to CSV now?",
    )
    if resp:
        exportar_indice_area_csv()

    if fig is not None:
        try:
            fig.close()
        except Exception:
            plt.close(fig)


def _ativar_seletor_elipse_indice(ax_zoom_img):
    """
    Enable elliptical selection on the zoomed index image.
    The resulting values (Npix,) are stored in indice_area_valores.
    After selecting, ask whether to save CSV and close the zoom window.
    """
    fig = ax_zoom_img.figure

    def _on_ellipse(ec, fc):
        global indice_zoomada, indice_area_valores

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

        if indice_zoomada is None:
            return

        h, w = indice_zoomada.shape
        Y, X = np.ogrid[:h, :w]
        mask = (((X - cx) / rx) ** 2 + ((Y - cy) / ry) ** 2) <= 1.0

        sel = indice_zoomada[mask]  # (Npix,)
        if sel.size == 0:
            return

        indice_area_valores = sel
        n = indice_area_valores.size

        resp = messagebox.askyesno(
            "Save index area data?",
            f"{n} pixels were selected on index '{indice_nome_atual}'.\n\n"
            "Do you want to export these values to CSV now?",
        )
        if resp:
            exportar_indice_area_csv()

        try:
            fig.close()
        except Exception:
            plt.close(fig)

    sel = EllipseSelector(
        ax_zoom_img,
        _on_ellipse,
        useblit=True,
        button=[1],
        interactive=True,
        props=dict(facecolor="none", edgecolor="yellow", linewidth=1.5),
    )
    _seletores_ativos_vi.append(sel)


def _ativar_seletor_poligono_indice(ax_zoom_img):
    """
    Enable polygon selection on the zoomed index image.

    Click to add vertices and double-click to close the polygon.
    Values are extracted for all pixels inside the polygon.
    After selecting, ask whether to save CSV and close the zoom window.
    """
    fig = ax_zoom_img.figure

    def _on_polygon(verts):
        global indice_zoomada, indice_area_valores

        if not verts or len(verts) < 3:
            return

        if indice_zoomada is None:
            return

        h, w = indice_zoomada.shape
        p = Path(verts)
        Y, X = np.mgrid[:h, :w]
        pts = np.vstack((X.ravel(), Y.ravel())).T
        mask = p.contains_points(pts).reshape(h, w)

        sel = indice_zoomada[mask]  # (Npix,)
        if sel.size == 0:
            return

        indice_area_valores = sel
        n = indice_area_valores.size

        resp = messagebox.askyesno(
            "Save index area data?",
            f"{n} pixels were selected on index '{indice_nome_atual}'.\n\n"
            "Do you want to export these values to CSV now?",
        )
        if resp:
            exportar_indice_area_csv()

        try:
            fig.close()
        except Exception:
            plt.close(fig)

    # PolygonSelector relies on Line2D → use color instead of face/edgecolor
    sel = PolygonSelector(
        ax_zoom_img,
        _on_polygon,
        useblit=True,
        props=dict(color="yellow", linewidth=1.5),
    )
    _seletores_ativos_vi.append(sel)


def on_select_zoom_indice(ecanto, fcanto):
    """
    RectangleSelector callback to define the zoom region for index-based analysis.
    After zooming, the user chooses the selection tool (rectangle, ellipse, or polygon)
    to extract values from a specific area within the zoomed index.
    """
    global indice_imagem_atual, indice_zoomada, indice_area_valores

    if indice_imagem_atual is None:
        print("[EasySpectra] Warning: no index image set for zoom.")
        return

    if ecanto.xdata is None or fcanto.xdata is None:
        return

    x1, y1 = int(ecanto.xdata), int(ecanto.ydata)
    x2, y2 = int(fcanto.xdata), int(fcanto.ydata)

    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    sub = indice_imagem_atual[ymin:ymax, xmin:xmax]
    if sub.size == 0:
        return

    indice_zoomada = sub.astype(float)

    # Normalization for display
    vmin = np.nanpercentile(indice_zoomada, 2)
    vmax = np.nanpercentile(indice_zoomada, 98)
    if vmax <= vmin:
        vmin = np.nanmin(indice_zoomada)
        vmax = np.nanmax(indice_zoomada)

    fig_zoom, ax_zoom = plt.subplots(figsize=(10, 8))
    im = ax_zoom.imshow(indice_zoomada, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax_zoom.set_title("2️⃣ Select an area on the zoomed index")
    ax_zoom.axis("off")
    plt.colorbar(im, ax=ax_zoom, fraction=0.046, pad=0.04)

    # Select area selection mode after zoom
    modo = _escolher_modo_selecao_indice()
    indice_area_valores = None

    _limpar_seletores_vi()
    if modo == "ret":
        sel = RectangleSelector(
            ax_zoom,
            on_select_area_indice_retangulo,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
            props=dict(facecolor="none", edgecolor="yellow", linewidth=1.5),
        )
        _seletores_ativos_vi.append(sel)
    elif modo == "circ":
        _ativar_seletor_elipse_indice(ax_zoom)
    else:  # "pol"
        _ativar_seletor_poligono_indice(ax_zoom)

    plt.tight_layout()
    plt.show()


def selecionar_area_sobre_indice():
    """
    High-level workflow for area-based index analysis:
    1) Display the full index image;
    2) User selects a zoom rectangle;
    3) Within the zoom, user chooses an area selection tool (rect/ellipse/polygon);
    4) Values are extracted for the selected area and stored in indice_area_valores;
    5) Immediately after selection, ask user if they want to export CSV.
    """
    global indice_imagem_atual

    if indice_imagem_atual is None:
        messagebox.showerror(
            "Error",
            "No vegetation index is currently selected. "
            "Please compute and choose an index first.",
        )
        return

    img = indice_imagem_atual
    if img.ndim != 2:
        messagebox.showerror(
            "Error",
            "The current index does not have a 2D shape (H, W).",
        )
        return

    vmin = np.nanpercentile(img, 2)
    vmax = np.nanpercentile(img, 98)
    if vmax <= vmin:
        vmin = np.nanmin(img)
        vmax = np.nanmax(img)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(img, cmap="RdYlGn", vmin=vmin, vmax=vmax)
    ax.set_title("1️⃣ Select a region to zoom in")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _limpar_seletores_vi()
    selector_zoom = RectangleSelector(
        ax,
        on_select_zoom_indice,
        useblit=True,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
        props=dict(facecolor="none", edgecolor="yellow", linewidth=1.5),
    )
    _seletores_ativos_vi.append(selector_zoom)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# TAB FACTORY
# ---------------------------------------------------------------------

def criar_aba_indices_vegetacao(aba):
    """
    Create the 'Vegetation indices' tab inside the provided parent widget.

    This tab allows the user to:
    - Import a cube (.npy) for vegetation indices;
    - Compute standard vegetation indices (NDVI, GNDVI, NDRE);
    - Create custom indices based on a free mathematical expression;
    - Visualize a selected index as an image;
    - Zoom and select an area (rectangle / ellipse / polygon).
      After selection, the user is asked if they want to export CSV;
    - Export selected area values to CSV (also available as explicit button).
    """
    frame = ttk.Frame(aba)
    frame.pack(anchor="w", padx=20, pady=20, fill="both", expand=True)

    # Status label (cube loaded / not loaded)
    status_label = tk.Label(
        frame,
        text="No spectral cube loaded.",
        font=("Arial", 10, "italic"),
    )
    status_label.pack(anchor="w", pady=(0, 8))

    def _atualizar_status():
        if cube_vi is not None and wavelengths_vi is not None:
            h, w, b = cube_vi.shape
            status_label.config(
                text=f"Cube loaded for vegetation indices: {w} x {h} px, {b} bands.",
            )
        else:
            status_label.config(text="No spectral cube loaded.")

    def _ui_importar_cubo():
        importar_cubo_para_indices()
        _atualizar_status()
        _atualizar_lista_indices()

    # Buttons row 1: import cube, compute predefined indices
    btn_frame1 = ttk.Frame(frame)
    btn_frame1.pack(anchor="w", pady=5, fill="x")

    tk.Button(
        btn_frame1,
        text="📂 Import cube (.npy) for vegetation indices",
        command=_ui_importar_cubo,
        font=("Arial", 11),
    ).pack(anchor="w", pady=2)

    def _ui_calcular_padrao():
        calcular_indices_padrao()
        _atualizar_lista_indices()

    tk.Button(
        btn_frame1,
        text="🌿 Compute standard indices (NDVI, GNDVI, NDRE)",
        command=_ui_calcular_padrao,
        font=("Arial", 11),
    ).pack(anchor="w", pady=2)

    # List of available indices
    tk.Label(
        frame,
        text="Available vegetation indices:",
        font=("Arial", 11, "bold"),
    ).pack(anchor="w", pady=(10, 2))

    listbox_indices = Listbox(
        frame,
        selectmode=tk.SINGLE,
        height=6,
        width=40,
    )
    listbox_indices.pack(anchor="w", pady=2, fill="x")

    def _atualizar_lista_indices():
        listbox_indices.delete(0, tk.END)
        for nome in sorted(indices_dict.keys()):
            listbox_indices.insert(tk.END, nome)

    # Buttons row 2: visualize, zoom/area, export CSV
    btn_frame2 = ttk.Frame(frame)
    btn_frame2.pack(anchor="w", pady=10, fill="x")

    def _get_indice_selecionado():
        sel = listbox_indices.curselection()
        if not sel:
            messagebox.showerror(
                "Index selection",
                "Please select an index from the list.",
            )
            return None
        return listbox_indices.get(sel[0])

    def _ui_visualizar():
        nome = _get_indice_selecionado()
        if nome is None:
            return
        visualizar_indice(nome)

    tk.Button(
        btn_frame2,
        text="👁️ Visualize selected index",
        command=_ui_visualizar,
        font=("Arial", 11),
    ).pack(anchor="w", pady=2)

    def _ui_selecionar_area():
        global indice_nome_atual, indice_imagem_atual
        nome = _get_indice_selecionado()
        if nome is None:
            return
        # Define o índice atual; a função de área vai abrir as janelas de zoom/seleção
        indice_nome_atual = nome
        indice_imagem_atual = indices_dict[nome]
        selecionar_area_sobre_indice()

    tk.Button(
        btn_frame2,
        text="🔍 Zoom + select area on index",
        command=_ui_selecionar_area,
        font=("Arial", 11),
    ).pack(anchor="w", pady=2)

    def _ui_exportar_csv():
        exportar_indice_area_csv()

    tk.Button(
        btn_frame2,
        text="💾 Export area index data to CSV",
        command=_ui_exportar_csv,
        font=("Arial", 11),
    ).pack(anchor="w", pady=2)

    # --- Custom index calculator ---
    tk.Label(
        frame,
        text="Custom index calculator:",
        font=("Arial", 11, "bold"),
    ).pack(anchor="w", pady=(15, 4))

    def _ui_custom_index():
        """
        Open a dialog that lets the user define a custom index:
        - Name
        - Free mathematical expression using b0, b1, ..., b(N-1)
        """
        if not _garantir_cubo_carregado():
            return
        _atualizar_status()

        win = Toplevel()
        win.title("Custom vegetation index")
        win.geometry("520x460")
        win.grab_set()

        tk.Label(
            win,
            text="Custom vegetation index",
            font=("Arial", 12, "bold"),
        ).pack(pady=6)

        # Index name
        tk.Label(win, text="Index name:").pack(anchor="w", padx=10)
        nome_var = tk.StringVar(value="MyIndex")
        tk.Entry(win, textvariable=nome_var).pack(
            anchor="w", padx=10, pady=(0, 6), fill="x"
        )

        # Bands list (index and wavelength)
        frame_lists = ttk.Frame(win)
        frame_lists.pack(anchor="w", padx=10, pady=4, fill="both", expand=True)

        tk.Label(
            frame_lists,
            text="Available bands (use b0, b1, ... in the expression):",
        ).grid(row=0, column=0, sticky="w")

        lista_bandas = Listbox(frame_lists, height=8)
        lista_bandas.grid(row=1, column=0, sticky="nsew", pady=2)
        frame_lists.rowconfigure(1, weight=1)
        frame_lists.columnconfigure(0, weight=1)

        if wavelengths_vi is not None:
            for i, wl in enumerate(wavelengths_vi):
                rotulo = f"b{i}: {wl:.1f} nm"
                lista_bandas.insert(tk.END, rotulo)

        # Expression field
        tk.Label(
            win,
            text="Expression (Python / NumPy syntax):",
        ).pack(anchor="w", padx=10, pady=(8, 2))

        expr_text = tk.Text(win, height=5)
        expr_text.pack(anchor="w", padx=10, pady=(0, 4), fill="both", expand=False)

        # Example expression
        expr_text.insert(
            "1.0",
            "(b8 - b3) / (b8 + b3 + 1e-12)",
        )

        # Helper text
        helper = (
            "Examples:\n"
            "  (b8 - b3) / (b8 + b3 + 1e-12)\n"
            "  np.log(b10 + 1) - np.log(b2 + 1)\n"
            "  (b5 - b4) / (b5 + b4 + 1e-12)\n\n"
            "Available functions: np, sin, cos, tan, exp, log, sqrt, abs, where."
        )
        tk.Label(win, text=helper, justify="left").pack(
            anchor="w", padx=10, pady=(0, 6)
        )

        def _confirmar():
            nome = nome_var.get().strip()
            if not nome:
                messagebox.showerror(
                    "Custom index",
                    "Please provide a name for the index.",
                )
                return

            expressao = expr_text.get("1.0", "end").strip()
            if not expressao:
                messagebox.showerror(
                    "Custom index",
                    "Please provide an expression.",
                )
                return

            try:
                criar_indice_personalizado(nome, expressao)
                _atualizar_lista_indices()
                win.destroy()
            except Exception as e:
                messagebox.showerror(
                    "Custom index",
                    f"Failed to create custom index:\n{type(e).__name__}: {e}",
                )

        tk.Button(
            win,
            text="Create index",
            command=_confirmar,
            font=("Arial", 11),
        ).pack(pady=10)

    tk.Button(
        frame,
        text="➕ Create custom vegetation index",
        command=_ui_custom_index,
        font=("Arial", 11),
    ).pack(anchor="w", pady=4)

    # Initial status and index list
    _atualizar_status()
    _atualizar_lista_indices()











