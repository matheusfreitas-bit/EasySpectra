# exploratoria_rgb.py
#
# Interactive RGB band exploration and zoom visualization for hyperspectral cubes.
# NOTE: Internal function names and workflow are preserved exactly.

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import numpy as np
import tkinter as tk
from tkinter import Toplevel, Listbox, Scrollbar, Button, messagebox

from .funcoes_importacao import get_cube, get_wavelengths


def visualizar_bandas_interativa():
    """
    Open an interactive window allowing the user to:
      1. Select 1–3 spectral bands from the loaded cube;
      2. Visualize an RGB (or grayscale) composite;
      3. Select an area with a rectangle and display a zoom window.

    User-facing text is fully translated to English, but internal logic and function
    names remain unchanged.
    """
    cube = get_cube()
    wavelengths = get_wavelengths()

    if cube is None or wavelengths is None:
        messagebox.showerror("Error", "No data has been loaded.")
        return

    # Band selection window
    janela_bandas = Toplevel()
    janela_bandas.title("Select bands for visualization")
    janela_bandas.geometry("320x400")

    # Scrollable frame
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

    # Populate list with wavelength labels
    for i, wl in enumerate(wavelengths):
        lista.insert(tk.END, f"{i}: {wl:.1f} nm")

    lista.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=lista.yview)

    def confirmar():
        """
        Confirm band selection, generate RGB visualization,
        and enable interactive zoom with RectangleSelector.
        """
        indices = lista.curselection()
        if len(indices) not in [1, 2, 3]:
            messagebox.showerror("Error", "Please select 1, 2, or 3 bands.")
            return

        bandas = list(indices)

        # Extract selected bands
        imagem = cube[:, :, bandas].astype(float)

        # Convert to RGB or grayscale according to the number of selected bands
        if len(bandas) == 1:
            imagem = imagem[:, :, 0]  # grayscale
        elif len(bandas) == 2:
            imagem = np.stack(
                [
                    imagem[:, :, 0],
                    imagem[:, :, 1],
                    np.zeros_like(imagem[:, :, 0]),
                ],
                axis=-1,
            )

        # Normalize for visualization (global normalization)
        imagem_norm = (imagem - imagem.min()) / (imagem.max() - imagem.min() + 1e-12)

        # Main display window
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(imagem_norm)
        ax.set_title(
            f"Bands: {[round(wavelengths[i], 1) for i in bandas]} (R-G-B composite)"
        )
        ax.axis("off")

        # Enable zoom of a selected rectangular area
        def onselect(ecanto, fcanto):
            """
            Callback for RectangleSelector.
            Extracts the selected region and displays a zoomed preview.
            """
            if ecanto.xdata is None or fcanto.xdata is None:
                return

            x1, y1 = int(ecanto.xdata), int(ecanto.ydata)
            x2, y2 = int(fcanto.xdata), int(fcanto.ydata)
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])

            if imagem_norm.ndim == 3:
                recorte = imagem_norm[ymin:ymax, xmin:xmax, :]
            else:
                recorte = imagem_norm[ymin:ymax, xmin:xmax]

            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.imshow(recorte)
            ax2.set_title("Zoom of selected area")
            ax2.axis("off")
            plt.tight_layout()
            plt.show()

        RectangleSelector(
            ax,
            onselect,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )

        plt.tight_layout()
        plt.show()
        janela_bandas.destroy()

    # Button to confirm visualization
    botao = Button(janela_bandas, text="Visualize", command=confirmar)
    botao.pack(pady=10)






