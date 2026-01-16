# interface.py
#
# EasySpectra main graphical user interface.
# NOTE: Function names and external API are preserved.
# Only UI texts, comments and documentation were adapted
# for an English, commercially-oriented version.

import os
import sys
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox, Toplevel, Listbox, Scrollbar, Button, Label
from PIL import Image, ImageTk

# IMPORTAÇÃO DAS FUNÇÕES EXTERNAS (nomes mantidos em português para compatibilidade)
from .funcoes_importacao import (
    cadastrar_bandas_gui,
    alinhar_a_partir_do_cadastro_gui,
    carregar_npy_com_interpolacao,
    alinhar_cubo_multimetodos_gui,   # alinhamento do cubo .npy / ortomosaico
    carregar_ortomosaico_gui,        # importar ortomosaico GeoTIFF/COG/JP2/VRT
)

from .preprocessamento import criar_aba_preprocessamento
from .analise_espectral import criar_aba_analise_espectral
from .geo_import import geoimport_wizard_gui
from .indices_vegetacao import criar_aba_indices_vegetacao  # NEW TAB


# ======== DEFAULT FONT CONFIGURATION ========
DEFAULT_FONT = ("Helvetica",)


def get_resource_path(relative_path: str) -> Path:
    """
    Return the absolute path to a resource (image, model, etc.),
    working both in development and when bundled with PyInstaller.

    Parameters
    ----------
    relative_path : str
        Relative path to the resource, from the project root in dev mode or
        from the PyInstaller bundle root when frozen.

    Returns
    -------
    pathlib.Path
        Absolute path to the requested resource.
    """
    try:
        # When bundled by PyInstaller
        base_path = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    except AttributeError:
        # Development mode:
        # interface.py is in src/easyspectra/
        # go two levels up -> project root (where logo.png usually is)
        base_path = Path(__file__).resolve().parents[2]

    return base_path / relative_path


def main():
    """
    Main entry point that creates and runs the EasySpectra graphical user interface.
    """

    # Main window
    janela = tk.Tk()
    # apply default font to all widgets
    janela.option_add("*Font", DEFAULT_FONT)
    janela.title("EasySpectra - Main Interface")
    janela.geometry("900x600")

    # Load and display the software logo (logo.png in project root or inside bundle)
    try:
        logo_path = get_resource_path("logo.png")
        imagem = Image.open(logo_path)
        imagem = imagem.resize((120, 120))
        imagem_tk = ImageTk.PhotoImage(imagem)
        label_imagem = tk.Label(janela, image=imagem_tk)
        # keep a reference to avoid being garbage-collected
        label_imagem.image = imagem_tk
        label_imagem.pack(anchor="w", padx=10, pady=5)
    except Exception as e:
        print(f"[EasySpectra] Error loading logo image: {e}")

    # Main notebook (tabs)
    abas = ttk.Notebook(janela)
    abas.pack(fill="both", expand=True)

    # -------------------------------
    # TAB 1 - IMPORT / REGISTRATION
    # -------------------------------
    aba_importacao = ttk.Frame(abas)
    abas.add(aba_importacao, text="Import")

    frame_import = ttk.Frame(aba_importacao)
    frame_import.pack(anchor="w", padx=20, pady=20)

    def ao_clicar_cadastrar():
        """
        Import non-georeferenced .tif bands and optionally run alignment afterwards.
        """
        # Importa bandas .tif SEM georreferência e oferece alinhar na sequência
        ok = cadastrar_bandas_gui()
        if not ok:
            return
        try:
            if messagebox.askyesno(
                "Run alignment now?",
                "Do you want to run the alignment for the imported .tif bands?",
            ):
                alinhar_a_partir_do_cadastro_gui()
                messagebox.showinfo("Alignment", "Alignment successfully completed.")
        except Exception as e:
            messagebox.showerror(
                "Alignment error",
                f"{type(e).__name__}: {e}",
            )

    btn_cadastrar = tk.Button(
        frame_import,
        text="Import non-georeferenced .tif bands",
        command=ao_clicar_cadastrar,
        relief="raised",
    )
    btn_cadastrar.pack(anchor="w", pady=5)

    def ao_clicar_npy():
        """
        Import a .npy spectral cube and configure band metadata
        (no interpolation/alignment prompts here).
        """
        caminho = filedialog.askopenfilename(
            title="Import .npy file",
            filetypes=[("NumPy array files", "*.npy")],
        )
        if not caminho:
            return
        # Define bands (Manual or Interval) and save JSON — no interpolation/alignment prompts
        carregar_npy_com_interpolacao(caminho)

    btn_npy = tk.Button(
        frame_import,
        text="Import .npy spectral cube",
        command=ao_clicar_npy,
        relief="raised",
    )
    btn_npy.pack(anchor="w", pady=5)

    # Import multiband orthomosaic (GeoTIFF / COG / JP2 / VRT)
    def ao_clicar_ortomosaico():
        """
        Import a multiband orthomosaic and optionally run fine alignment
        to improve band-to-band overlay.
        """
        try:
            carregar_ortomosaico_gui()
            # After importing the multiband orthomosaic, offer fine alignment
            if messagebox.askyesno(
                "Run fine alignment?",
                "Do you want to run fine alignment to improve band-to-band overlap "
                "of the imported orthomosaic?",
            ):
                alinhar_cubo_multimetodos_gui()
                messagebox.showinfo(
                    "Alignment",
                    "Orthomosaic fine alignment successfully completed.",
                )
        except Exception as e:
            messagebox.showerror("Error", f"{type(e).__name__}: {e}")

    btn_ortho = tk.Button(
        frame_import,
        text="Import orthomosaic (GeoTIFF / COG / JP2)",
        command=ao_clicar_ortomosaico,
        relief="raised",
    )
    btn_ortho.pack(anchor="w", pady=5)

    # GeoImport button (ODM + orthomosaic creation)
    btn_geo = tk.Button(
        frame_import,
        text="GeoImport: import georeferenced .tif bands and create orthomosaic",
        command=lambda: (
            geoimport_wizard_gui(),
            (
                messagebox.askyesno(
                    "Run alignment now?",
                    "Do you want to align the newly created orthomosaic?",
                )
                and not alinhar_cubo_multimetodos_gui() is None
            )
            or True,
        ),
        relief="raised",
    )
    btn_geo.pack(anchor="w", pady=5)

    # -------------------------------
    # TAB 2 - ALIGNMENT
    # -------------------------------
    aba_alinhamento = ttk.Frame(abas)
    abas.add(aba_alinhamento, text="Alignment")

    frame_alinh = ttk.Frame(aba_alinhamento)
    frame_alinh.pack(anchor="w", padx=20, pady=20)

    def _ui_btn_alinhar_cadastro():
        """
        Run alignment using previously registered non-georeferenced .tif bands.
        """
        try:
            alinhar_a_partir_do_cadastro_gui()
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to align from registered bands (.tif): {e}",
            )

    btn_alinhar_cadastro = tk.Button(
        frame_alinh,
        text="Align and stack imported .tif bands",
        command=_ui_btn_alinhar_cadastro,
        relief="raised",
    )
    btn_alinhar_cadastro.pack(anchor="w", pady=5)

    def _ui_btn_alinhar_cubo():
        """
        Run alignment for a spectral cube or orthomosaic (.npy / GeoTIFF).
        """
        try:
            alinhar_cubo_multimetodos_gui()
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to align cube/orthomosaic: {e}",
            )

    btn_alinhar_cubo = tk.Button(
        frame_alinh,
        text="Align cube/orthomosaic (.npy / GeoTIFF)",
        command=_ui_btn_alinhar_cubo,
        relief="raised",
    )
    btn_alinhar_cubo.pack(anchor="w", pady=5)

    # -------------------------------
    # TAB 3 - PRE-PROCESSING
    # -------------------------------
    aba_preprocessamento = ttk.Frame(abas)
    abas.add(aba_preprocessamento, text="Pre-processing")
    criar_aba_preprocessamento(aba_preprocessamento)

    # -------------------------------
    # TAB 4 - SPECTRAL ANALYSIS
    # -------------------------------
    aba_analise = ttk.Frame(abas)
    abas.add(aba_analise, text="Spectral Analysis")
    criar_aba_analise_espectral(aba_analise)

    # -------------------------------
    # TAB 5 - VEGETATION INDICES
    # -------------------------------
    aba_indices = ttk.Frame(abas)
    abas.add(aba_indices, text="Vegetation indices")
    criar_aba_indices_vegetacao(aba_indices)

    # Start application
    janela.mainloop()


if __name__ == "__main__":
    main()






