# preprocessamento.py
"""
Preprocessing tab for EasySpectra.

This module creates the GUI tab that handles spectral cube preprocessing
(normalization and smoothing). It allows the user to:

    - Select up to four normalization/smoothing methods in sequence.
    - Apply them to the currently loaded cube.
    - Save the normalized cube as a .npy file, together with a JSON file
      containing the wavelength metadata.
    - Load a previously normalized cube (.npy) and its associated wavelengths
      JSON, or reconstruct wavelengths via a user-defined min/max range.

Important:
    - Internal method identifiers (e.g., "minmax_banda", "nenhum") are kept
      for compatibility with `normalizar_cubo`.
    - All user-facing text (labels, dialogs, messages) is in English.
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import numpy as np
import os
import json

from .funcoes_importacao import get_cube, set_cube, get_wavelengths, set_wavelengths
from .normalizacao import normalizar_cubo  # Normalization function


# Internal method identifiers (must remain unchanged)
opcoes = [
    "nenhum",
    "minmax_banda",
    "minmax_global",
    "zscore",
    "zscore_global",
    "snv_local",
    "sg_smoothing",
]

# User-facing labels for the normalization methods
_METHOD_LABELS = {
    "nenhum": "None",
    "minmax_banda": "Min–max per band",
    "minmax_global": "Min–max (global)",
    "zscore": "Z-score per band",
    "zscore_global": "Z-score (global)",
    "snv_local": "SNV per spectrum (pixel)",
    "sg_smoothing": "Savitzky–Golay smoothing",
}

# Inverse mapping: label → internal code
_LABEL_TO_CODE = {v: k for k, v in _METHOD_LABELS.items()}


def criar_aba_preprocessamento(aba):
    """
    Create the preprocessing tab inside the main EasySpectra interface.

    Parameters
    ----------
    aba : tkinter.Widget
        Parent notebook/tab where the preprocessing frame will be attached.
    """
    frame = ttk.Frame(aba)
    frame.pack(anchor="w", padx=20, pady=20)

    ttk.Label(
        frame,
        text="Choose up to 4 normalization methods:",
        font=("Arial", 12),
    ).pack(anchor="w")

    # Comboboxes for selecting up to 4 methods in sequence
    dropdowns = []
    combo_values = [_METHOD_LABELS[cod] for cod in opcoes]
    none_label = _METHOD_LABELS["nenhum"]

    for _ in range(4):
        cb = ttk.Combobox(
            frame,
            values=combo_values,
            state="readonly",
            width=30,
        )
        cb.set(none_label)
        cb.pack(anchor="w", pady=2)
        dropdowns.append(cb)

    def executar_normalizacao():
        """
        Apply the selected normalization methods to the current cube and save
        the result (.npy + .json with wavelengths).
        """
        selected_labels = [cb.get() for cb in dropdowns if cb.get() != none_label]
        metodos = [_LABEL_TO_CODE[label] for label in selected_labels]

        if not metodos:
            messagebox.showwarning(
                "Warning",
                "Please select at least one normalization method.",
            )
            return

        cubo = get_cube()
        wavelengths = get_wavelengths()
        if cubo is None or wavelengths is None:
            messagebox.showerror(
                "Error",
                "No data has been imported. Please import a cube before preprocessing.",
            )
            return

        cubo_norm = normalizar_cubo(cubo, metodo=metodos)

        nome_arquivo = simpledialog.askstring(
            "Save as",
            "Name of the new normalized cube file (without extension):",
        )
        if not nome_arquivo:
            return

        caminho = filedialog.asksaveasfilename(
            defaultextension=".npy",
            initialfile=nome_arquivo,
            filetypes=[("NumPy files", "*.npy")],
        )
        if caminho:
            # Save normalized cube
            np.save(caminho, cubo_norm)

            # Save wavelength metadata as JSON
            json_path = caminho.replace(".npy", ".json")
            with open(json_path, "w") as f:
                json.dump({"wavelengths": wavelengths.tolist()}, f, indent=4)

            messagebox.showinfo(
                "Success",
                (
                    f"Normalized cube saved to:\n{caminho}\n\n"
                    f"Wavelength metadata saved to:\n{json_path}"
                ),
            )

    def carregar_normalizado():
        """
        Load a previously normalized cube (.npy) and its wavelength metadata.

        The function tries to find a JSON file with the same base name to
        restore wavelengths. If not found or unreadable, it asks the user
        for minimum and maximum wavelength values and builds a linear range.
        """
        caminho = filedialog.askopenfilename(
            title="Select normalized cube (.npy)",
            filetypes=[("NumPy files", "*.npy")],
        )
        if not caminho:
            return

        cubo = np.load(caminho)
        set_cube(cubo)

        # Attempt to find a JSON metadata file containing wavelengths
        json_path = caminho.replace(".npy", ".json")
        if os.path.exists(json_path):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
                wavelengths = np.array(data["wavelengths"])
                set_wavelengths(wavelengths)
                messagebox.showinfo(
                    "Import completed",
                    "Normalized cube and wavelength metadata successfully loaded "
                    "from JSON file.",
                )
                return
            except Exception as e:
                messagebox.showwarning(
                    "Error reading JSON",
                    f"An error occurred while loading wavelength metadata:\n{e}",
                )

        # Fallback: manual wavelength definition
        comprimento_min = simpledialog.askinteger(
            "Minimum wavelength",
            "Enter the minimum wavelength (e.g., 400):",
            initialvalue=400,
        )
        comprimento_max = simpledialog.askinteger(
            "Maximum wavelength",
            "Enter the maximum wavelength (e.g., 1000):",
            initialvalue=1000,
        )
        if comprimento_min is None or comprimento_max is None:
            messagebox.showwarning(
                "Cancelled",
                "Import cancelled by user.",
            )
            return

        bandas = cubo.shape[2]
        wavelengths = np.linspace(comprimento_min, comprimento_max, bandas)
        set_wavelengths(wavelengths)
        messagebox.showinfo(
            "Import completed",
            "Normalized cube loaded.\nWavelengths have been generated using a "
            "linear interpolation between the specified minimum and maximum values.",
        )

    # Buttons
    tk.Button(
        frame,
        text="Load normalized cube (.npy)",
        command=carregar_normalizado,
        font=("Arial", 12),
        relief="raised",
    ).pack(anchor="w", pady=5)

    tk.Button(
        frame,
        text="Apply normalization and save",
        command=executar_normalizacao,
        font=("Arial", 12),
        relief="raised",
    ).pack(anchor="w", pady=10)




