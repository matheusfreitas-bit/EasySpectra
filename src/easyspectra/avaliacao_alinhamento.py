# avaliacao_alinhamento.py
#
# Alignment quality assessment for pairwise band registration.
# Compares multiple alignment methods using SSIM and reports the best one.

import os  # required for os.path.basename
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


def comparar_multiplos_metodos(arquivos=None, bandas_nm=None, idx_ref=None):
    """
    Compare multiple alignment methods using SSIM.

    If `arquivos`, `bandas_nm` and `idx_ref` are provided, they are used directly
    (no interactive dialogs are opened).

    Otherwise, the function:
      1. Asks the user to select multiple .tif images (one per band);
      2. Asks the user to enter the wavelength (nm) for each selected image;
      3. Asks the user to choose a reference band;
      4. Applies all configured alignment methods and computes SSIM;
      5. Displays a bar chart with SSIM per method and indicates the best one.
    """
    from tkinter import filedialog, simpledialog, messagebox
    from tifffile import imread
    from metodos_alinhamento import alinhar_imagem

    # -------------------------
    # INPUT LOGIC
    # -------------------------
    if not arquivos or not bandas_nm or idx_ref is None:
        arquivos, bandas_nm = [], []

        # Step 1–2: select TIFFs and register wavelengths
        while True:
            caminho = filedialog.askopenfilename(
                title="Select .tif image",
                filetypes=[("TIFF files", "*.tif")],
            )
            if not caminho:
                break

            try:
                nm = simpledialog.askinteger(
                    "Wavelength (nm)",
                    f"Enter the wavelength (nm) for:\n{os.path.basename(caminho)}"
                )
                if nm is None:
                    continue
                arquivos.append(caminho)
                bandas_nm.append(nm)
            except Exception as e:
                messagebox.showerror(
                    "Error",
                    f"Failed to register band metadata:\n{e}",
                )

        if not arquivos or not bandas_nm:
            messagebox.showwarning("Cancelled", "No band was selected.")
            return

        # Step 3: choose reference band
        try:
            import tkinter as tk
            from tkinter import ttk, Toplevel, StringVar, Button

            win = Toplevel()
            win.title("Choose reference band")
            win.geometry("360x140")
            win.grab_set()

            tk.Label(win, text="Select the reference band (nm):").pack(pady=8)

            opcoes = [f"{i} — {int(b)} nm" for i, b in enumerate(bandas_nm)]
            var = StringVar(value=opcoes[0] if opcoes else "")
            combo = ttk.Combobox(
                win, values=opcoes, textvariable=var, state="readonly"
            )
            if opcoes:
                combo.current(0)
            combo.pack(pady=6, fill="x", padx=12)

            resultado = {"idx": None}

            def confirmar():
                sel = combo.get()
                if sel:
                    try:
                        resultado["idx"] = int(sel.split("—")[0].strip())
                    except Exception:
                        resultado["idx"] = 0
                win.destroy()

            Button(win, text="Confirm", command=confirmar).pack(pady=10)
            win.wait_window()
            idx_ref = resultado["idx"]

        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to select reference band:\n{e}",
            )
            return

        if idx_ref is None:
            messagebox.showwarning("Cancelled", "No reference band was selected.")
            return

    # -------------------------
    # PREPARATION
    # -------------------------
    if len(arquivos) < 2:
        messagebox.showwarning(
            "Attention",
            "Select at least two bands for comparison.",
        )
        return

    from tifffile import imread

    try:
        imagem_referencia = imread(arquivos[idx_ref])
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Unable to read reference image:\n{e}",
        )
        return

    # Choose a target band distinct from the reference (preserves original logic)
    alvo_idx = 0 if idx_ref != 0 else 1
    try:
        imagem_alvo_base = imread(arquivos[alvo_idx])
    except Exception as e:
        messagebox.showerror(
            "Error",
            f"Unable to read target image:\n{e}",
        )
        return

    # Ensure single-channel for SSIM
    def to_gray(img):
        if img.ndim == 3:
            try:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            except Exception:
                return img.mean(axis=2)
        return img

    ref = to_gray(imagem_referencia).astype(np.float32)

    # Internal method IDs — DO NOT CHANGE
    metodos = ["orb", "ecc", "template", "akaze", "sift", "mi", "superglue"]

    # User-friendly labels for visualization
    metodo_labels = {
        "orb": "ORB keypoints",
        "ecc": "ECC (intensity-based)",
        "template": "Template matching",
        "akaze": "AKAZE keypoints",
        "sift": "SIFT keypoints",
        "mi": "Mutual information",
        "superglue": "SuperGlue (deep matching)",
    }

    # -------------------------
    # EXECUTION
    # -------------------------
    resultados = []
    for metodo in metodos:
        try:
            print(f"[EasySpectra] Testing alignment method: {metodo}")
            alinhado = alinhar_imagem(imagem_referencia, imagem_alvo_base, metodo=metodo)
            aligned_gray = to_gray(alinhado).astype(np.float32)

            # Match dimensions if needed
            if aligned_gray.shape != ref.shape:
                aligned_gray = cv2.resize(
                    aligned_gray,
                    (ref.shape[1], ref.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

            vmin = float(min(ref.min(), aligned_gray.min()))
            vmax = float(max(ref.max(), aligned_gray.max()))
            dr = max(vmax - vmin, 1e-6)

            score_ssim = ssim(ref, aligned_gray, data_range=dr)
            resultados.append((metodo, score_ssim))

        except Exception as e:
            print(f"[EasySpectra] Error with method {metodo}: {e}")
            continue

    if not resultados:
        messagebox.showerror(
            "Error",
            "No valid result was obtained from the alignment methods.",
        )
        return

    # -------------------------
    # OUTPUT
    # -------------------------
    resultados.sort(key=lambda x: x[1], reverse=True)
    melhor_metodo, melhor_score = resultados[0]

    plt.figure(figsize=(8, 4))
    metodos_ids, scores = zip(*resultados)
    metodos_lbl = [metodo_labels.get(m, m.upper()) for m in metodos_ids]
    plt.bar(metodos_lbl, scores)
    plt.title("Alignment Method Comparison (SSIM)")
    plt.ylabel("SSIM")
    plt.xlabel("Method")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.show()

    best_label = metodo_labels.get(melhor_metodo, melhor_metodo.upper())
    messagebox.showinfo(
        "Best method",
        f"The best alignment method was:\n{best_label}\n(SSIM = {melhor_score:.4f})",
    )


