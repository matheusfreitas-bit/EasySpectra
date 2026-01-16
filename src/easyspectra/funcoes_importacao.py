# -*- coding: utf-8 -*-
# funcoes_importacao.py — version with visible ROI fix and more robust panel alignment
from tkinter import messagebox

import os
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, Toplevel, Button, Label, StringVar, Text
from tkinter import ttk
from tifffile import imread
from skimage.metrics import structural_similarity as ssim
from .metodos_alinhamento import alinhar_imagem


# ---- Reentrancy guard for auto-alignment to avoid recursion depth in Tk callbacks
_ALIGN_RUNNING = False
def _try_align_imediato(align_callable, warn_prefix="Alignment"):
    """Executes align_callable once, deferring to the Tk event loop and guarding reentrancy.
    This avoids recursive re-entry (maximum recursion depth) when called inside other GUI handlers.
    """
    import tkinter as _tk
    from tkinter import messagebox as _mb
    global _ALIGN_RUNNING
    if _ALIGN_RUNNING:
        return
    _ALIGN_RUNNING = True
    def _run():
        global _ALIGN_RUNNING
        try:
            align_callable()
        except RecursionError as e:
            _mb.showwarning(warn_prefix, f"Failed to start immediate alignment: {e}")
        except Exception as e:
            _mb.showwarning(warn_prefix, f"Failed to start immediate alignment: {e}")
        finally:
            _ALIGN_RUNNING = False
    root = _tk._default_root
    if root is None:
        try:
            root = _tk.Tk()
            root.withdraw()
        except Exception:
            # last resort: run directly
            _run()
            return
    # Defer to next tick to escape current callback stack
    root.after(50, _run)


# Global variables
cube = None
wavelengths = None

# In-memory registration (fast session persistence)
_cadastro_atual = {"arquivos": [], "bandas_nm": []}


def _ui_pergunta_rho_por_banda():
    from tkinter import simpledialog, messagebox
    win = Toplevel()
    win.title("Reflectance per band")
    win.geometry("520x220")
    win.grab_set()
    var = tk.IntVar(value=1)
    ttk.Label(
        win,
        text="Do you have reflectance values for the panel for EACH BAND?"
    ).pack(pady=10)
    ttk.Radiobutton(
        win,
        text="Yes — I will provide the reflectance (0–1) for each band",
        variable=var,
        value=1
    ).pack(anchor="w", padx=16)
    ttk.Radiobutton(
        win,
        text="No — use a SINGLE global value (not recommended)",
        variable=var,
        value=0
    ).pack(anchor="w", padx=16)
    btnframe = ttk.Frame(win)
    btnframe.pack(pady=12)
    escolha = {"val": None}
    def ok():
        escolha["val"] = bool(var.get())
        win.destroy()
    def cancelar():
        escolha["val"] = None
        win.destroy()
    ttk.Button(btnframe, text="Continue", command=ok).pack(side="left", padx=6)
    ttk.Button(btnframe, text="Cancel", command=cancelar).pack(side="left", padx=6)
    win.wait_window()
    return escolha["val"]


# -------------------------
# UI utilities (scroll)
# -------------------------
def _make_scrolled_form(title="Parameters", size="480x500"):
    win = Toplevel()
    win.title(title)
    win.geometry(size)
    win.grab_set()
    canvas = tk.Canvas(win)
    scroll_y = tk.Scrollbar(win, orient="vertical", command=canvas.yview)
    frame_scroll = tk.Frame(canvas)
    frame_scroll.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=frame_scroll, anchor="nw")
    canvas.configure(yscrollcommand=scroll_y.set)
    canvas.pack(side="left", fill="both", expand=True)
    scroll_y.pack(side="right", fill="y")
    return win, frame_scroll

def _add_row(frame, texto, key, defaults, rows, kind="entry", options=None, width=22, wrap=440):
    Label(frame, text=texto, anchor="w", justify="left", wraplength=wrap).pack(pady=(8, 0), anchor="w")
    if kind == "entry":
        var = StringVar(value=str(defaults[key]))
        e = tk.Entry(frame, textvariable=var, width=width)
        e.pack(anchor="w")
        rows.append((key, var, "entry"))
    elif kind == "combo":
        var = StringVar(value=str(defaults[key]))
        cb = ttk.Combobox(frame, values=options, state="readonly", width=min(width, 28))
        cb.set(str(defaults[key]))
        cb.pack(anchor="w")
        rows.append((key, cb, "combo"))

# -------------------------
# Band registration (import/label and save)
# -------------------------
def selecionar_arquivos_banda():
    arquivos_bandas, bandas_nm = [], []
    while True:
        caminho = filedialog.askopenfilename(
            title="Select .tif band file",
            filetypes=[("TIFF files", "*.tif *.tiff")]
        )
        if not caminho:
            break
        try:
            nm = simpledialog.askinteger(
                "Identify band",
                f"What is the wavelength (nm) of the file:\n{os.path.basename(caminho)}?"
            )
            if nm is None:
                continue
            arquivos_bandas.append(caminho)
            bandas_nm.append(nm)
        except Exception as e:
            messagebox.showerror("Error", f"Error adding band: {e}")
        if not messagebox.askyesno("Add another band?", "Do you want to add another band?"):
            break
    return arquivos_bandas, bandas_nm

def _salvar_cadastro_em(caminho_json, arquivos, bandas_nm):
    dados = {"arquivos": arquivos, "bandas_nm": bandas_nm}
    with open(caminho_json, "w") as f:
        json.dump(dados, f, indent=4)


def cadastrar_bandas_gui_legacy():
    """
    Opens a Tkinter WINDOW to register .tif bands without georeference.
    - Lists files and corresponding wavelengths (nm) in a Treeview.
    - Buttons: Add, Edit nm, Remove, Clear, Sort by nm, Save and use, Cancel.
    - On save: sorts by nm, updates _cadastro_atual and ALWAYS saves a JSON in the
      directory of the first file.
    """
    import os, json, re
    import tkinter as _tk
    from tkinter import filedialog as _fd, messagebox as _mb, simpledialog as _sd, ttk as _ttk
    import numpy as _np

    global _cadastro_atual

    win = _tk.Toplevel()
    win.title("Band registration (.tif without georeference)")
    win.geometry("800x560")
    win.grab_set()

    # Treeview (file, nm)
    frame = _tk.Frame(win)
    frame.pack(fill="both", expand=True, padx=14, pady=(12, 8))

    cols = ("arquivo", "nm")
    tv = _ttk.Treeview(frame, columns=cols, show="headings", selectmode="extended")
    tv.heading("arquivo", text="File (.tif)")
    tv.heading("nm", text="Wavelength (nm)")
    tv.column("arquivo", width=560, anchor="w")
    tv.column("nm", width=160, anchor="center")

    sb = _tk.Scrollbar(frame, orient="vertical", command=tv.yview)
    tv.configure(yscrollcommand=sb.set)
    tv.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    def _parse_nm_from_name(name):
        # tries to extract patterns like '750nm' or '_750_' from filename
        m = re.search(r'(\d{3,4})\s*nm', name.lower())
        if m:
            try:
                return int(m.group(1))
            except:
                pass
        m = re.search(r'[_\-](\d{3,4})(?:[_\-.]|$)', name.lower())
        if m:
            try:
                return int(m.group(1))
            except:
                pass
        return None

    def _adicionar():
        paths = _fd.askopenfilenames(
            title="Select band .tif files",
            filetypes=[("TIFF files", "*.tif *.tiff")]
        )
        if not paths:
            return
        for p in paths:
            nm_guess = _parse_nm_from_name(os.path.basename(p))
            if nm_guess is None:
                try:
                    nm_guess = _sd.askinteger(
                        "Wavelength (nm)",
                        f"Provide nm for:\n{os.path.basename(p)}",
                        minvalue=200, maxvalue=2500
                    )
                except Exception:
                    nm_guess = None
            if nm_guess is None:
                # If user cancels, still insert empty for later editing
                tv.insert("", "end", values=(p, ""))
            else:
                tv.insert("", "end", values=(p, int(nm_guess)))

    def _editar_nm():
        sel = tv.selection()
        if not sel:
            _mb.showwarning("No selection", "Select a row to edit nm.")
            return
        for iid in sel:
            path_atual, nm_atual = tv.item(iid, "values")
            try:
                novo_nm = _sd.askinteger(
                    "Edit nm",
                    f"Provide nm for:\n{os.path.basename(path_atual)}",
                    initialvalue=(int(nm_atual) if str(nm_atual).strip().isdigit() else None),
                    minvalue=200, maxvalue=2500
                )
            except Exception:
                novo_nm = None
            if novo_nm is not None:
                tv.item(iid, values=(path_atual, int(novo_nm)))

    def _remover():
        sel = tv.selection()
        for iid in sel:
            tv.delete(iid)

    def _limpar():
        for iid in tv.get_children():
            tv.delete(iid)

    def _ordenar_por_nm():
        itens = [(iid, tv.item(iid, "values")) for iid in tv.get_children()]
        def _nm_val(v):
            try:
                return int(v[1])
            except:
                return 10**9
        itens.sort(key=lambda t: _nm_val(t[1]))
        _limpar()
        for _, vals in itens:
            tv.insert("", "end", values=vals)

    def _salvar_e_usar():
        # collect rows
        itens = [tv.item(iid, "values") for iid in tv.get_children()]
        if not itens:
            _mb.showwarning("Empty", "Add at least one band.")
            return
        # validate nms
        arquivos = []
        bandas_nm = []
        for (p, nm) in itens:
            if not p or not os.path.exists(p):
                _mb.showerror("Invalid file", f"File not found:\n{p}")
                return
            try:
                nm_int = int(nm)
            except Exception:
                _mb.showerror("Invalid nm", f"Provide nm for:\n{os.path.basename(p)}")
                return
            arquivos.append(p)
            bandas_nm.append(nm_int)

        # sort by nm
        pares = sorted(zip(bandas_nm, arquivos), key=lambda x: x[0])
        bandas_nm = [p[0] for p in pares]
        arquivos   = [p[1] for p in pares]

        # update state in memory
        global _cadastro_atual
        _cadastro_atual = {"arquivos": arquivos, "bandas_nm": bandas_nm}

        # mandatory JSON save
        diretorio = os.path.dirname(arquivos[0]) if arquivos else os.getcwd()
        base = "cadastro_bandas"
        destino = os.path.join(diretorio, f"{base}.json")
        if os.path.exists(destino):
            i = 1
            while True:
                cand = os.path.join(diretorio, f"{base}({i}).json")
                if not os.path.exists(cand):
                    destino = cand
                    break
                i += 1
        try:
            with open(destino, "w", encoding="utf-8") as f:
                json.dump({"arquivos": arquivos, "bandas_nm": bandas_nm}, f, indent=4, ensure_ascii=False)
            _cadastro_atual["path_json"] = destino
            _mb.showinfo("Registration saved", f"Registration saved at:\n{destino}")
            win.destroy()
        except Exception as e:
            _mb.showerror("Save error", f"Could not save registration (.json):\n{e}")

    # Buttons
    btns = _tk.Frame(win)
    btns.pack(fill="x", padx=14, pady=(4, 12))

    _tk.Button(btns, text="Add bands", command=_adicionar).pack(side="left", padx=4)
    _tk.Button(btns, text="Edit nm", command=_editar_nm).pack(side="left", padx=4)
    _tk.Button(btns, text="Remove", command=_remover).pack(side="left", padx=4)
    _tk.Button(btns, text="Clear", command=_limpar).pack(side="left", padx=4)
    _tk.Button(btns, text="Sort by nm", command=_ordenar_por_nm).pack(side="left", padx=12)

    _tk.Button(btns, text="Save and use", command=_salvar_e_usar).pack(side="right", padx=4)
    _tk.Button(btns, text="Cancel", command=win.destroy).pack(side="right", padx=4)


def carregar_cadastro_gui():
    global _cadastro_atual
    caminho = filedialog.askopenfilename(
        title="Load registration (.json)",
        filetypes=[("Registration JSON", "*.json")]
    )
    if not caminho:
        return
    try:
        with open(caminho, "r") as f:
            dados = json.load(f)
        arquivos = dados.get("arquivos", [])
        bandas_nm = dados.get("bandas_nm", [])
        if not arquivos or not bandas_nm or len(arquivos) != len(bandas_nm):
            messagebox.showerror("Error", "Invalid or incomplete registration.")
            return
        _cadastro_atual = {"arquivos": arquivos, "bandas_nm": bandas_nm}
        messagebox.showinfo("Registration loaded", f"{len(arquivos)} bands loaded from registration.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load registration: {e}")

# -------------------------
# Reference choices and methods
# -------------------------


# === Reusable helper: selects .tif files per band using the SAME window
#     as the 'Import .tif without georeference' dialog
def selecionar_tifs_por_banda_via_cadastro():
    """
    Opens the same interactive window used to register bands (.tif without georeference)
    and returns (arquivos, bandas_nm) based on _cadastro_atual.
    Does not change external logic: only encapsulates collection.
    """
    global _cadastro_atual
    # open window so the user can build/edit the list
    try:
        cadastrar_bandas_gui()
    except Exception as _e:
        try:
            from tkinter import messagebox as _mb
            _mb.showerror("Error", f"Failed to open band registration window:\n{type(_e).__name__}: {_e}")
        except Exception:
            pass
        return [], []
    arquivos = list(_cadastro_atual.get("arquivos", []) or [])
    bandas_nm = list(_cadastro_atual.get("bandas_nm", []) or [])
    # safety: keeps pairing and ordering as done in registration
    if len(arquivos) != len(bandas_nm) or not arquivos:
        try:
            from tkinter import messagebox as _mb
            _mb.showwarning("Incomplete registration", "No valid band was selected.")
        except Exception:
            pass
        return [], []
    return arquivos, bandas_nm

def escolher_banda_referencia(bandas_nm):
    win = Toplevel()
    win.title("Choose reference band")
    win.geometry("360x140")
    win.grab_set()
    tk.Label(win, text="Select the reference band (nm):").pack(pady=8)
    opcoes = [f"{i} — {int(b)} nm" for i, b in enumerate(bandas_nm)]
    var = StringVar(value=opcoes[0] if opcoes else "")
    combo = ttk.Combobox(win, values=opcoes, textvariable=var, state="readonly")
    if opcoes:
        combo.current(0)
    combo.pack(pady=6, fill="x", padx=12)
    resultado = {"idx": None}
    def confirmar():
        sel = combo.get()
        try:
            idx = int(sel.split(" — ")[0].strip())
        except Exception:
            idx = None
        resultado["idx"] = idx
        win.destroy()
    Button(win, text="Confirm", command=confirmar).pack(pady=8)
    win.wait_window()
    return resultado["idx"]

def escolher_metodos_alinhamento():
    metodos = ["orb", "sift", "akaze", "ecc", "template", "mi", "superglue"]
    nomes = {
        "orb": "ORB keypoints",
        "sift": "SIFT keypoints",
        "akaze": "AKAZE keypoints",
        "ecc": "ECC (intensity-based)",
        "template": "Template matching",
        "mi": "Mutual information",
        "superglue": "SuperGlue (deep matching)"
    }
    win = Toplevel()
    win.title("Select alignment methods")
    win.geometry("360x360")
    win.grab_set()
    Label(win, text="Select the methods you want to apply:").pack(pady=8)
    vars_ = {}
    for m in metodos:
        v = tk.IntVar(value=0)
        tk.Checkbutton(win, text=nomes[m], variable=v, anchor="w").pack(fill="x", padx=12)
        vars_[m] = v
    resultado = {"selecionados": []}
    def confirmar():
        sel = [m for m, v in vars_.items() if v.get() == 1]
        if not sel:
            messagebox.showwarning("Warning", "Select at least one method.")
            return
        resultado["selecionados"] = sel
        win.destroy()
    Button(win, text="Confirm", command=confirmar).pack(pady=10)
    win.wait_window()
    return resultado["selecionados"]

def coletar_parametros_para_metodos(metodos):
    params_por_metodo = {}
    name2func = {
        "superglue": globals().get("pedir_parametros_superglue"),
        "orb": globals().get("pedir_parametros_orb"),
        "sift": globals().get("pedir_parametros_sift"),
        "ecc": globals().get("pedir_parametros_ecc"),
        "template": globals().get("pedir_parametros_template"),
        "akaze": globals().get("pedir_parametros_akaze"),
        "mi": globals().get("pedir_parametros_mi"),
    }
    for m in metodos:
        f = name2func.get(m)
        params_por_metodo[m] = f() if callable(f) else {}
    return params_por_metodo

# ---- SSIM helper (normalizes only for SSIM; final cube remains raw)
def _ssim_norm(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a_min, a_max = float(a.min()), float(a.max())
    b_min, b_max = float(b.min()), float(b.max())
    a_n = (a - a_min) / (a_max - a_min + 1e-12)
    b_n = (b - b_min) / (b_max - b_min + 1e-12)
    return float(ssim(a_n, b_n, data_range=1.0))

def _mostrar_resultados_ssim(resultados_por_metodo):
    win, frame = _make_scrolled_form("SSIM results (by method)", "520x520")
    txt = Text(frame, height=28, wrap="word")
    txt.pack(fill="both", expand=True)
    for metodo, info in resultados_por_metodo.items():
        media = info.get("media")
        txt.insert("end", f"=== {metodo.upper()} ===\n")
        if "scores" in info and info["scores"]:
            for nm, val in info["scores"]:
                txt.insert("end", f"• {nm} nm: SSIM = {'n/a' if val is None else f'{val:.3f}'}\n")
        txt.insert("end", f"Mean SSIM (no ref): {'n/a' if media is None else f'{media:.3f}'}\n\n")
    Button(frame, text="Close", command=win.destroy).pack(pady=8)
    win.wait_window()

def _escolher_metodo_prosseguir(metodos_disponiveis):
    win = Toplevel()
    win.title("Proceed with method")
    win.geometry("360x140")
    win.grab_set()
    Label(win, text="Choose the method to save the cube:").pack(pady=8)
    nomes = {
        "orb": "ORB keypoints",
        "sift": "SIFT keypoints",
        "akaze": "AKAZE keypoints",
        "ecc": "ECC (intensity-based)",
        "template": "Template matching",
        "mi": "Mutual information",
        "superglue": "SuperGlue (deep matching)"
    }
    opcoes = [f"{m} — {nomes[m]}" for m in metodos_disponiveis]
    var = StringVar(value=opcoes[0] if opcoes else "")
    combo = ttk.Combobox(win, values=opcoes, textvariable=var, state="readonly")
    if opcoes:
        combo.current(0)
    combo.pack(pady=6, fill="x", padx=12)
    resultado = {"metodo": None}
    def confirmar():
        sel = combo.get()
        try:
            metodo = sel.split(" — ")[0].strip()
        except Exception:
            metodo = None
        resultado["metodo"] = metodo
        win.destroy()
    Button(win, text="Confirm", command=confirmar).pack(pady=10)
    win.wait_window()
    return resultado["metodo"]

# -------------------------
# NEW: Alignment from registration
# -------------------------
def alinhar_a_partir_do_cadastro_gui():
    global cube, wavelengths, _cadastro_atual

    arquivos = _cadastro_atual.get("arquivos", [])
    bandas_nm = _cadastro_atual.get("bandas_nm", [])
    
    # Safety: if there is no registration in memory, try to reload from recent JSON
    if (not arquivos or not bandas_nm) and _cadastro_atual.get("path_json"):
        try:
            with open(_cadastro_atual["path_json"], "r", encoding="utf-8") as _f:
                _dados = json.load(_f)
            arquivos = _dados.get("arquivos", []) or arquivos
            bandas_nm = _dados.get("bandas_nm", []) or bandas_nm
        except Exception:
            pass
    if not arquivos or not bandas_nm:
        messagebox.showwarning(
            "No registration",
            "No registration found in memory. Load a registration (.json) in the Alignment tab."
        )
        return

    # If it is a single multiband file
    if len(arquivos) == 1:
        img = imread(arquivos[0])
        if img.ndim == 3:
            cube = img.astype(np.float32)
            wavelengths = np.array(bandas_nm)
            messagebox.showinfo(
                "Import finished",
                "Multiband cube loaded without the need for alignment."
            )
        else:
            messagebox.showwarning("Error", "Selected image is not a valid multiband cube.")
        return

    idx_ref = escolher_banda_referencia(bandas_nm)
    if idx_ref is None:
        messagebox.showwarning("Cancelled", "No reference band was selected.")
        return

    metodos_escolhidos = escolher_metodos_alinhamento()
    if not metodos_escolhidos:
        messagebox.showwarning("Cancelled", "No method was selected.")
        return

    params_por_metodo = coletar_parametros_para_metodos(metodos_escolhidos)

    # Load images
    bandas_imagens = [imread(c) for c in arquivos]
    imagem_referencia = bandas_imagens[idx_ref]

    resultados_por_metodo = {}
    cubos_por_metodo = {}

    for metodo in metodos_escolhidos:
        print(f"▶️ Method: {metodo}")
        params = params_por_metodo.get(metodo, {})
        bandas_alinhadas = []
        ssim_scores = []
        for i, img in enumerate(bandas_imagens):
            if i == idx_ref:
                bandas_alinhadas.append(img)
            else:
                print(f"   - Aligning band {bandas_nm[i]} nm...")
                img_alinhada = alinhar_imagem(imagem_referencia, img, metodo=metodo, params=params)
                bandas_alinhadas.append(img_alinhada)
                try:
                    ssim_val = _ssim_norm(imagem_referencia, img_alinhada)
                except Exception:
                    ssim_val = None
                ssim_scores.append((bandas_nm[i], ssim_val))

        cubo_m = np.stack(bandas_alinhadas, axis=-1).astype(np.float32)
        cubos_por_metodo[metodo] = cubo_m
        vals = [v for (_, v) in ssim_scores if v is not None]
        media = np.mean(vals) if vals else None
        resultados_por_metodo[metodo] = {"scores": ssim_scores, "media": media}

    _mostrar_resultados_ssim(resultados_por_metodo)

    metodo_final = _escolher_metodo_prosseguir(metodos_escolhidos)
    if not metodo_final:
        messagebox.showinfo("Cancelled", "Operation cancelled before saving.")
        return

    cube = cubos_por_metodo[metodo_final]
    wavelengths = np.array(bandas_nm)

    # >>> PANEL CALIBRATION (optional) — WITHOUT GEOREF
    if messagebox.askyesno(
        "Panel-based correction",
        "Do you want to apply radiometric correction using panels (white/gray/black)?"
    ):
        try:
            cube_corr, detalhes_cal = (
                lambda _cube, _waves: (
                    calibrar_cubo_por_paineis_com_bandas_gui(_cube, _waves)
                    if _ui_modo_calibracao_paineis() == 'por_banda'
                    else calibrar_cubo_por_paineis_foto_unica_gui(_cube, _waves)
                )
            )(cube, wavelengths)
            if detalhes_cal is not None:
                cube = cube_corr  # use calibrated cube
        except Exception as e:
            messagebox.showwarning(
                "Panel-based correction",
                f"Panel correction failed:\n{e}\nProceeding without correction."
            )

    nome_base = simpledialog.askstring("Save as", "Base name to save aligned cube (without extension):")
    if not nome_base:
        return

    diretorio = os.path.dirname(arquivos[0])
    caminho_npy = os.path.join(diretorio, f"{nome_base}.npy")
    caminho_json = os.path.join(diretorio, f"{nome_base}.json")

    np.save(caminho_npy, cube)
    with open(caminho_json, 'w') as f:
        json.dump({"wavelengths": wavelengths.tolist()}, f, indent=4)

    messagebox.showinfo(
        "Import finished",
        f"Cube saved with method '{metodo_final.upper()}' at:\n{caminho_npy}\n\n"
        f"Bands at:\n{caminho_json}"
    )

# -------------------------
# Import .npy with interpolation (unchanged behavior)
# -------------------------


def _ui_escolha_bandas_para_npy(num_bandas):
    win = Toplevel()
    win.title("Band identification")
    win.geometry("560x320")
    win.grab_set()
    modo = tk.StringVar(value="intervalo")
    ttk.Label(
        win,
        text=f"The cube has {num_bandas} bands. How do you want to define the wavelengths?"
    ).pack(pady=10, padx=16, anchor="w")
    frm_rad = ttk.Frame(win)
    frm_rad.pack(fill="x", padx=16)
    ttk.Radiobutton(
        frm_rad,
        text="By interval (min and max)",
        variable=modo,
        value="intervalo"
    ).pack(anchor="w")
    ttk.Radiobutton(
        frm_rad,
        text="Manual (provide each nm)",
        variable=modo,
        value="manual"
    ).pack(anchor="w")

    frm_int = ttk.LabelFrame(win, text="Interval")
    frm_int.pack(fill="x", padx=16, pady=(8, 4))
    ttk.Label(frm_int, text="Min (nm):").grid(row=0, column=0, sticky="w", padx=(4, 6), pady=4)
    ent_min = ttk.Entry(frm_int, width=12)
    ent_min.grid(row=0, column=1, sticky="w", pady=4)
    ttk.Label(frm_int, text="Max (nm):").grid(row=0, column=2, sticky="w", padx=(12, 6), pady=4)
    ent_max = ttk.Entry(frm_int, width=12)
    ent_max.grid(row=0, column=3, sticky="w", pady=4)
    ttk.Label(
        frm_int,
        text="Bands will be distributed uniformly between min and max."
    ).grid(row=1, column=0, columnspan=4, sticky="w", padx=(4, 6), pady=(0, 6))

    frm_man = ttk.LabelFrame(win, text="Manual")
    frm_man.pack(fill="both", expand=True, padx=16, pady=(4, 8))
    ttk.Label(frm_man, text="Type wavelengths (nm) separated by commas:").pack(
        anchor="w", padx=8, pady=(6, 2)
    )
    txt = tk.Text(frm_man, height=4)
    txt.pack(fill="both", expand=True, padx=8, pady=(0, 6))

    vals = {"ok": False}

    def _ok():
        m = modo.get()
        try:
            if m == "intervalo":
                mi = float(ent_min.get())
                ma = float(ent_max.get())
                if not (ma > mi):
                    raise ValueError
                import numpy as _np
                seq = list(_np.round(_np.linspace(mi, ma, num_bandas), 6))
                vals["modo"] = "intervalo"
                vals["lista"] = seq
            else:
                lista = [
                    float(x.strip())
                    for x in txt.get("1.0", "end").strip().split(",")
                    if x.strip()
                ]
                if len(lista) != num_bandas:
                    messagebox.showerror(
                        "Incompatible bands",
                        f"You provided {len(lista)} wavelengths, "
                        f"but the cube has {num_bandas} bands."
                    )
                    return
                vals["modo"] = "manual"
                vals["lista"] = lista
        except Exception:
            messagebox.showerror("Error", "Check the values provided.")
            return
        vals["ok"] = True
        win.destroy()

    ttk.Button(win, text="OK", command=_ok).pack(pady=6)
    win.wait_window()
    return vals if vals["ok"] else None

def carregar_npy_com_interpolacao(caminho_npy, comprimento_min=400, comprimento_max=1000):
    # NOTE: This function does not perform automatic interpolation or trigger alignment.
    # It only asks the user how to define the bands (Manual or Interval),
    # and saves a JSON file with the wavelengths.
    global cube, wavelengths
    cube = np.load(caminho_npy)
    num_bandas = cube.shape[2]

    cfg = _ui_escolha_bandas_para_npy(num_bandas)
    if not cfg:
        messagebox.showwarning("Cancelled", "Import was cancelled.")
        return

    wavelengths = np.array(cfg["lista"], dtype=float)

    json_path = caminho_npy.replace(".npy", ".json")
    try:
        with open(json_path, 'w') as f:
            json.dump({"wavelengths": wavelengths.tolist()}, f, indent=4)
    except Exception as e:
        messagebox.showwarning("Warning", f"Could not save wavelengths JSON: {e}")
    else:
        messagebox.showinfo(
            "Import finished",
            f"Cube imported and wavelengths defined ({cfg['modo']}).\n\n"
            f"Cube: {caminho_npy}\nJSON: {json_path}"
        )

    return cube, wavelengths


def get_cube():
    return cube


def get_wavelengths():
    return wavelengths


def set_cube(novo_cubo):
    global cube
    cube = novo_cubo


def set_wavelengths(novas_wavelengths):
    global wavelengths
    wavelengths = novas_wavelengths


# =============================================================================
#  IMPORT ORTHOMOSAIC (GeoTIFF/COG/JP2/VRT) + PREVIEW + ORDER + SAVE NPY
# =============================================================================

def _inferir_wavelengths_rasterio(ds):
    """Attempts to infer wavelengths from:
    1) Per-band tags (WAVELENGTH, wavelength, CENTRAL_WAVELENGTH, WVL, BAND_WAVELENGTH)
    2) Per-band description (ds.descriptions[b-1], looking for '### nm')
    3) Per-band tag 'DESCRIPTION' when present
    4) Dataset tag 'BAND_DESCRIPTIONS' separated by '|', looking for '### nm'
    Returns a list of floats or None.
    """
    waves = []
    # Steps 1–3: per band
    for b in range(1, ds.count + 1):
        cand = None
        tags = {}
        try:
            tags = ds.tags(b) or {}
        except Exception:
            tags = {}
        # 1) wavelength keys
        for k in ["WAVELENGTH", "wavelength", "CENTRAL_WAVELENGTH", "WVL", "BAND_WAVELENGTH"]:
            if k in tags:
                cand = tags[k]
                break
        # 2) description via ds.descriptions
        if cand is None:
            try:
                desc2 = ds.descriptions[b - 1] or ""
            except Exception:
                desc2 = ""
            if desc2:
                import re as _re
                m2 = _re.search(r"(\d{3,4})\s*nm", str(desc2))
                if m2:
                    cand = m2.group(1)
        # 3) DESCRIPTION inside band tags
        if cand is None:
            desc = tags.get("DESCRIPTION") or tags.get("DESC") or ""
            if desc:
                import re as _re
                m = _re.search(r"(\d{3,4})\s*nm", desc)
                if m:
                    cand = m.group(1)

        if cand is None:
            waves = None
            break
        try:
            waves = (waves or [])
            waves.append(float(str(cand).replace(",", ".").strip()))
        except Exception:
            waves = None
            break

    if waves is not None and len(waves) == ds.count:
        return waves

    # Step 4: BAND_DESCRIPTIONS at dataset level
    try:
        dstags = ds.tags() or {}
        bdesc = (
            dstags.get("BAND_DESCRIPTIONS")
            or dstags.get("BandDescriptions")
            or dstags.get("band_descriptions")
        )
        if bdesc:
            parts = [p.strip() for p in str(bdesc).split("|") if p.strip()]
            found = []
            import re as _re
            for p in parts:
                m = _re.search(r"(\d{3,4})\s*nm", p)
                if m:
                    found.append(float(m.group(1)))
            if len(found) == ds.count:
                return found
    except Exception:
        pass

    return None


def _selecionar_bandas_rgb_por_waves(waves):
    import numpy as _np
    target_rgb = [650, 560, 480]
    arr = _np.array(waves, dtype=float).reshape(-1)
    idxs = []
    for t in target_rgb:
        dif = _np.abs(arr - t)
        idxs.append(int(dif.argmin()))
    return tuple(idxs)


def gerar_preview_ortomosaico(caminho_tif, max_side=1024):
    import rasterio
    import numpy as _np

    with rasterio.open(caminho_tif) as ds:
        ratio = max(
            ds.height / max_side if ds.height > max_side else 1.0,
            ds.width / max_side if ds.width > max_side else 1.0,
        )
        if ratio > 1.0:
            new_h = max(1, int(ds.height / ratio))
            new_w = max(1, int(ds.width / ratio))
            data = ds.read(
                out_shape=(ds.count, new_h, new_w),
                resampling=rasterio.enums.Resampling.bilinear,
            )
        else:
            data = ds.read()
        cube_hw = _np.moveaxis(data, 0, -1)  # (B,H,W)->(H,W,B)
        B = cube_hw.shape[2]
        waves = _inferir_wavelengths_rasterio(ds)

        if B >= 3:
            if waves is not None and len(waves) == B:
                r, g, b = _selecionar_bandas_rgb_por_waves(waves)
            else:
                r, g, b = (0, 1, 2)
            rgb = cube_hw[:, :, [r, g, b]].astype("float32")
            for i in range(3):
                ch = rgb[:, :, i]
                minv, maxv = _np.nanpercentile(ch, 2), _np.nanpercentile(ch, 98)
                if maxv > minv:
                    rgb[:, :, i] = (ch - minv) / (maxv - minv)
            return rgb
        else:
            ch = cube_hw[:, :, 0].astype("float32")
            minv, maxv = _np.nanpercentile(ch, 2), _np.nanpercentile(ch, 98)
            if maxv > minv:
                ch = (ch - minv) / (maxv - minv)
            return _np.dstack([ch, ch, ch])


def carregar_ortomosaico_geotiff(
    caminho_tif,
    usar_nan_para_nodata=True,
    downsample_factor=None,
    target_max_size=None
):
    import rasterio
    import numpy as np

    with rasterio.open(caminho_tif) as ds:
        if downsample_factor and downsample_factor > 1:
            new_h = max(1, ds.height // int(downsample_factor))
            new_w = max(1, ds.width // int(downsample_factor))
            data = ds.read(
                out_shape=(ds.count, new_h, new_w),
                resampling=rasterio.enums.Resampling.bilinear,
            )
        elif target_max_size and isinstance(target_max_size, (list, tuple)) and len(target_max_size) == 2:
            max_h, max_w = target_max_size
            ratio_h = ds.height / max_h if ds.height > max_h else 1.0
            ratio_w = ds.width / max_w if ds.width > max_w else 1.0
            ratio = max(ratio_h, ratio_w)
            if ratio > 1.0:
                new_h = max(1, int(ds.height / ratio))
                new_w = max(1, int(ds.width / ratio))
                data = ds.read(
                    out_shape=(ds.count, new_h, new_w),
                    resampling=rasterio.enums.Resampling.bilinear,
                )
            else:
                data = ds.read()
        else:
            data = ds.read()

        nodata = ds.nodata
        try:
            mask = ds.read_masks(1)
        except Exception:
            mask = None

        cube_hw = np.moveaxis(data, 0, -1)  # (B,H,W)->(H,W,B)

        if usar_nan_para_nodata:
            if mask is not None:
                cube_hw = cube_hw.astype("float32", copy=False)
                cube_hw[mask == 0] = np.nan
            if nodata is not None:
                cube_hw = cube_hw.astype("float32", copy=False)
                cube_hw[cube_hw == nodata] = np.nan

        waves = _inferir_wavelengths_rasterio(ds)
        profile = ds.profile.copy()

    if waves is None:
        waves = list(range(1, cube_hw.shape[2] + 1))

    try:
        _waves = np.array(waves, dtype=float).reshape(-1)
        if _waves.size == cube_hw.shape[2]:
            order = np.argsort(_waves)
            if not np.all(order == np.arange(_waves.size)):
                cube_hw = cube_hw[:, :, order]
                _waves = _waves[order]
            waves = _waves
    except Exception:
        pass

    set_cube(cube_hw)
    set_wavelengths(np.array(waves, dtype=float))
    return cube_hw, np.array(waves, dtype=float), profile


def salvar_cubo_npy(caminho_base=None):
    import numpy as _np
    import json as _json
    from tkinter import filedialog as _fd, messagebox as _mb

    cube = get_cube()
    waves = get_wavelengths()
    if cube is None or waves is None:
        _mb.showwarning("Nothing to save", "Cube or wavelengths not defined.")
        return None, None

    if caminho_base is None:
        caminho_base = _fd.asksaveasfilename(
            title="Save cube (.npy) and wavelengths (.json)",
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy")],
        )
        if not caminho_base:
            return None, None
        if caminho_base.lower().endswith(".npy"):
            caminho_base = caminho_base[:-4]

    npy_path = caminho_base + ".npy"
    json_path = caminho_base + ".json"

    _np.save(npy_path, cube)
    with open(json_path, "w") as f:
        _json.dump({"wavelengths": [float(x) for x in waves]}, f, indent=2)

    try:
        _mb.showinfo("Saved", f"Files saved:\n{npy_path}\n{json_path}")
    except Exception:
        pass
    return npy_path, json_path


def carregar_ortomosaico_gui():
    from tkinter import filedialog as _fd, messagebox as _mb, simpledialog as _sd
    import os as _os
    import numpy as _np

    caminho = _fd.askopenfilename(
        title="Import orthomosaic (GeoTIFF/COG/JP2/VRT)",
        filetypes=[("Rasters", "*.tif *.tiff *.vrt *.jp2"), ("All files", "*.*")],
    )
    if not caminho:
        return

    reduzir = _mb.askyesno(
        "Reduce resolution?",
        "Do you want to reduce the resolution to save memory?"
    )
    fator = None
    max_side = None
    if reduzir:
        modo = _mb.askyesno(
            "Choose method",
            "Do you want to provide a factor (e.g. 4) instead of a target size?\n"
            "Yes = Factor, No = Maximum side size."
        )
        if modo:
            fator = _sd.askinteger(
                "Downsample factor",
                "Provide an integer factor (e.g. 2, 4, 8):",
                initialvalue=4,
                minvalue=2,
                maxvalue=64,
            )
        else:
            max_side = _sd.askinteger(
                "Maximum size (px)",
                "Provide the desired maximum side size (e.g. 2048):",
                initialvalue=2048,
                minvalue=256,
                maxvalue=16384,
            )

    ver_preview = _mb.askyesno("Preview", "Generate a quick RGB preview before loading the full cube?")
    if ver_preview:
        try:
            rgb = gerar_preview_ortomosaico(caminho, max_side=1024)
            import matplotlib.pyplot as _plt
            _plt.figure(figsize=(6, 4))
            _plt.imshow(rgb)
            _plt.title("RGB preview (auto-stretch)")
            _plt.axis("off")
            _plt.show(block=False)
        except Exception as _e:
            _mb.showwarning(
                "Preview failed",
                f"Could not generate preview.\n{type(_e).__name__}: {_e}",
            )

    kwargs = {}
    if fator:
        kwargs["downsample_factor"] = fator
    if max_side:
        kwargs["target_max_size"] = (max_side, max_side)

    try:
        cube_hw, waves, profile = carregar_ortomosaico_geotiff(caminho, **kwargs)
    except Exception as e:
        _mb.showerror("Import error", f"{type(e).__name__}: {e}")
        return

    placeholder = (
        getattr(waves, "dtype", None) is not None
        and waves.dtype.kind in "iu"
        and len(waves) == cube_hw.shape[2]
        and waves.min() == 1
    )
    if placeholder:
        resp = _mb.askyesno(
            "Map wavelengths",
            "The file did not provide wavelengths.\n"
            "Do you want to map them now (recommended)?"
        )
        if resp:
            try:
                from geo_import import map_bands_to_nm_ui
                novo = map_bands_to_nm_ui(list(range(1, cube_hw.shape[2] + 1)))
                if novo and isinstance(novo, (list, tuple)) and len(novo) == cube_hw.shape[2]:
                    waves = _np.array(novo, dtype=float)
            except Exception:
                comprimento_min = _sd.askinteger(
                    "Minimum wavelength (nm)",
                    "Type the minimum wavelength (e.g. 400):",
                    initialvalue=400,
                )
                comprimento_max = _sd.askinteger(
                    "Maximum wavelength (nm)",
                    "Type the maximum wavelength (e.g. 1000):",
                    initialvalue=1000,
                )
                if comprimento_min and comprimento_max:
                    waves = _np.linspace(comprimento_min, comprimento_max, cube_hw.shape[2])

    # >>> PANEL CALIBRATION (optional) — GEOREF (loader on this tab)
    if _mb.askyesno(
        "Panel-based correction",
        "Do you want to apply radiometric correction using panels (white/gray/black) before saving?"
    ):
        try:
            cube_corr, detalhes_cal = (
                lambda _cube, _waves: (
                    calibrar_cubo_por_paineis_com_bandas_gui(_cube, _waves)
                    if _ui_modo_calibracao_paineis() == 'por_banda'
                    else calibrar_cubo_por_paineis_foto_unica_gui(_cube, _waves)
                )
            )(get_cube(), get_wavelengths())
            if detalhes_cal is not None:
                set_cube(cube_corr)
        except Exception as _e:
            _mb.showwarning(
                "Panel-based correction",
                f"Panel correction failed:\n{type(_e).__name__}: {_e}\n"
                "Proceeding without correction."
            )

    set_wavelengths(waves)
    try:
        _mb.showinfo(
            "Import finished",
            f"Orthomosaic imported successfully!\n"
            f"Dimensions: {get_cube().shape} (H, W, B)"
        )
    except Exception:
        pass

    salvar = _mb.askyesno(
        "Save as .npy",
        "Do you want to save the cube and wavelengths for later use (.npy + .json)?"
    )
    if salvar:
        try:
            salvar_cubo_npy(None)
        except Exception as _e:
            _mb.showwarning("Save failed", f"{type(_e).__name__}: {_e}")

    return get_cube(), get_wavelengths()


# =============================================================================
#  BLOCK: Align multiband .npy cube + export multiband GeoTIFF
# =============================================================================

def alinhar_cubo_multimetodos_gui(caminho_npy=None):
    if caminho_npy is None:
        caminho_npy = filedialog.askopenfilename(
            title="Select the multiband .npy (geo_import)",
            filetypes=[("NPY files", "*.npy")],
        )
        if not caminho_npy:
            return
    base_path = os.path.splitext(caminho_npy)[0]
    try:
        cube_local = np.load(caminho_npy).astype(np.float32)
    except Exception as e:
        messagebox.showerror("Error", f"Could not load NPY: {e}")
        return

    json_path = base_path + ".json"
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                wavelengths_local = np.array(json.load(f).get("wavelengths", []), dtype=float)
        except Exception:
            wavelengths_local = None
    else:
        wavelengths_local = None

    if wavelengths_local is None or len(wavelengths_local) != cube_local.shape[2]:
        comprimento_min = simpledialog.askinteger(
            "Minimum wavelength",
            "Type the minimum wavelength (e.g. 400):",
            initialvalue=400,
        )
        comprimento_max = simpledialog.askinteger(
            "Maximum wavelength",
            "Type the maximum wavelength (e.g. 1000):",
            initialvalue=1000,
        )
        if comprimento_min is None or comprimento_max is None:
            messagebox.showwarning("Cancelled", "Operation cancelled.")
            return
        wavelengths_local = np.linspace(comprimento_min, comprimento_max, cube_local.shape[2])

    H, W, B = cube_local.shape
    if B < 2:
        messagebox.showwarning("Warning", "Cube has fewer than 2 bands.")
        return

    idx_ref = escolher_banda_referencia(wavelengths_local)
    if idx_ref is None:
        return
    metodos = escolher_metodos_alinhamento()
    if not metodos:
        messagebox.showwarning("Warning", "No method selected.")
        return

    ref = cube_local[:, :, idx_ref]
    resultados = []
    aligned_by_method = {}

    for metodo in metodos:
        alinhadas = []
        ssim_b = []
        for b in range(B):
            img = cube_local[:, :, b]
            if b == idx_ref:
                alinhadas.append(img)
                continue
            try:
                aligned = alinhar_imagem(ref, img, metodo=metodo)
                if aligned.shape != (H, W):
                    import cv2
                    aligned = cv2.resize(aligned, (W, H), interpolation=cv2.INTER_NEAREST)
                alinhadas.append(aligned)
                ssim_b.append(_ssim_norm(ref, aligned))
            except Exception as e:
                print(f"[{metodo}] error on band {b}: {e}")
                alinhadas.append(img)
                ssim_b.append(np.nan)

        cube_al = np.stack(alinhadas, axis=-1).astype(np.float32)
        aligned_by_method[metodo] = cube_al
        vals = [v for v in ssim_b if np.isfinite(v)]
        media = float(np.mean(vals)) if vals else float("nan")
        resultados.append((metodo, media, ssim_b))

    resultados.sort(key=lambda x: (x[1] if np.isfinite(x[1]) else -1), reverse=True)

    resultados_dict = {}
    for metodo, media, _ in resultados:
        s_list = []
        for b in range(B):
            if b == idx_ref:
                continue
            nm = wavelengths_local[b]
            try:
                s_val = _ssim_norm(ref, aligned_by_method[metodo][:, :, b])
            except Exception:
                s_val = None
            s_list.append((nm, s_val))
        resultados_dict[metodo] = {"scores": s_list, "media": media}

    _mostrar_resultados_ssim(resultados_dict)

    metodo_vencedor = _escolher_metodo_prosseguir([m for (m, _, __) in resultados])
    if not metodo_vencedor:
        messagebox.showinfo("Cancelled", "Operation cancelled before saving.")
        return

    out_base = f"{base_path}_aligned_{metodo_vencedor}"
    try:
        np.save(out_base + ".npy", aligned_by_method[metodo_vencedor])
        with open(out_base + ".json", "w") as f:
            json.dump({"wavelengths": list(map(float, wavelengths_local))}, f, indent=2)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save aligned files: {e}")
        return

    messagebox.showinfo(
        "Done",
        f"Aligned cube saved:\n{out_base}.npy\n{out_base}.json"
    )

    if messagebox.askyesno(
        "Export GeoTIFF?",
        "Do you also want to export an aligned multiband GeoTIFF?"
    ):
        tif_original = filedialog.askopenfilename(
            title="Select the original multiband (georeferenced) TIF",
            filetypes=[("GeoTIFF", "*.tif *.tiff")],
        )
        if tif_original:
            try:
                salvar_geotiff_alinhado_from_npy(out_base + ".npy", tif_original)
                messagebox.showinfo(
                    "GeoTIFF exported",
                    f"Saved at:\n{out_base}.tif"
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export GeoTIFF: {e}")


def salvar_geotiff_alinhado_from_npy(aligned_npy_path, original_tif_path, out_tif_path=None, dtype="float32"):
    import rasterio
    cube_al = np.load(aligned_npy_path).astype(np.float32)

    with rasterio.open(original_tif_path) as src:
        profile = src.profile.copy()
        profile.update(count=cube_al.shape[2], dtype=dtype)

        if out_tif_path is None:
            out_tif_path = aligned_npy_path.replace(".npy", ".tif")

        with rasterio.open(out_tif_path, "w", **profile) as dst:
            for i in range(cube_al.shape[2]):
                dst.write(cube_al[:, :, i].astype(dtype), i + 1)

    return out_tif_path


# ======================================================================
#  PANEL/TILE CALIBRATION (Empirical Line) — GUI
#  (with robust panel alignment and visible ROI)
# ======================================================================

def _ui_escolher_paineis_com_rho():
    win = Toplevel()
    win.title("Panel correction")
    win.geometry("420x260")
    win.grab_set()
    tk.Label(win, text="Do you want to use panels/tiles to correct the bands?").pack(pady=(10, 6))
    ans = {"ok": False}
    tk.Button(
        win,
        text="Yes",
        width=10,
        command=lambda: (ans.update(ok=True), win.destroy())
    ).pack(side="left", padx=12, pady=8)
    tk.Button(
        win,
        text="No",
        width=10,
        command=win.destroy
    ).pack(side="left", padx=12, pady=8)
    win.wait_window()
    if not ans["ok"]:
        return None

    win2 = Toplevel()
    win2.title("Select panels")
    win2.geometry("480x260")
    win2.grab_set()
    tk.Label(
        win2,
        text="Select the panels and set the nominal reflectance (0–1):"
    ).pack(pady=8)

    opts = {
        "white": {"checked": tk.IntVar(value=1), "rho": tk.DoubleVar(value=0.99)},
        "gray": {"checked": tk.IntVar(value=1), "rho": tk.DoubleVar(value=0.18)},
        "black": {"checked": tk.IntVar(value=0), "rho": tk.DoubleVar(value=0.03)},
    }
    for nome, cfg in opts.items():
        fr = tk.Frame(win2)
        fr.pack(fill="x", padx=12, pady=4)
        tk.Checkbutton(fr, text=nome.capitalize(), variable=cfg["checked"]).pack(side="left")
        tk.Label(fr, text="Reflectance:").pack(side="left", padx=6)
        tk.Entry(fr, textvariable=cfg["rho"], width=8).pack(side="left")

    done = {"ok": False}
    tk.Button(win2, text="Continue", command=lambda: (done.update(ok=True), win2.destroy())).pack(pady=10)
    win2.wait_window()
    if not done["ok"]:
        return None

    paineis = []
    for nome, cfg in opts.items():
        if cfg["checked"].get() == 1:
            try:
                rho = float(cfg["rho"].get())
            except Exception:
                rho = 0.0
            rho = max(0.0, min(1.0, rho))
            paineis.append((nome, rho))
    return paineis


def _carregar_painel_como_cubo(arquivos_por_banda):
    from tifffile import imread as _tf_imread
    imgs = []
    for p in arquivos_por_banda:
        arr = _tf_imread(p)
        if arr.ndim != 2:
            if arr.ndim == 3:
                arr = arr[0] if arr.shape[0] < min(arr.shape[1:]) else arr[:, :, 0]
            else:
                raise ValueError("Unexpected format for panel image.")
        imgs.append(arr.astype(np.float32))
    shapes = {img.shape for img in imgs}
    if len(shapes) != 1:
        raise ValueError(f"Panel images have different sizes: {shapes}")
    stack = np.stack(imgs, axis=-1).astype(np.float32)  # (H,W,B)
    return stack


# --------- ROBUST PANEL STACK ALIGNMENT: phase + ECC pyramid ----------
def _estimate_translation_phase(ref, mov):
    import cv2 as _cv2
    import numpy as _np
    # Hanning window to reduce border effects
    win = _np.outer(_np.hanning(ref.shape[0]), _np.hanning(ref.shape[1])).astype(_np.float32)
    r = _cv2.normalize(ref.astype(_np.float32), None, 0, 1, _cv2.NORM_MINMAX)
    m = _cv2.normalize(mov.astype(_np.float32), None, 0, 1, _cv2.NORM_MINMAX)
    r *= win
    m *= win
    (shift_y, shift_x), _ = _cv2.phaseCorrelate(r, m)
    return float(shift_x), float(shift_y)


def _alinhar_stack_painel(stack_in):
    import numpy as _np
    import cv2 as _cv2

    H, W, B = stack_in.shape
    ref = stack_in[:, :, 0].astype(_np.float32)

    def _norm01(img):
        vmin, vmax = _np.nanmin(img), _np.nanmax(img)
        if not _np.isfinite(vmin) or not _np.isfinite(vmax) or vmax <= vmin:
            return img.astype(_np.float32)
        return ((img - vmin) / (vmax - vmin + 1e-12)).astype(_np.float32)

    ref_n = _norm01(ref)
    out = _np.zeros_like(stack_in, dtype=_np.float32)
    out[:, :, 0] = stack_in[:, :, 0].astype(_np.float32)

    # ECC params
    warp_mode = _cv2.MOTION_AFFINE
    criteria = (_cv2.TERM_CRITERIA_EPS | _cv2.TERM_CRITERIA_COUNT, 200, 1e-6)

    for k in range(1, B):
        mov = stack_in[:, :, k].astype(_np.float32)
        mov_n = _norm01(mov)

        # (1) pre-alignment by phase correlation (translation)
        tx, ty = 0.0, 0.0
        try:
            tx, ty = _estimate_translation_phase(ref_n, mov_n)  # dx, dy
        except Exception:
            pass
        warp_matrix = _np.array([[1, 0, tx], [0, 1, ty]], dtype=_np.float32)

        try:
            # (2) refinement with ECC (pyramid)
            # downsample for stability (up to ~512px on largest side)
            scale = max(H, W) / 512.0
            if scale > 1.0:
                small_ref = _cv2.resize(ref_n, (int(W / scale), int(H / scale)), interpolation=_cv2.INTER_AREA)
                small_mov = _cv2.resize(mov_n, (int(W / scale), int(H / scale)), interpolation=_cv2.INTER_AREA)
                wm_small = warp_matrix.copy()
                wm_small[0, 2] /= scale
                wm_small[1, 2] /= scale
                _ = _cv2.findTransformECC(small_ref, small_mov, wm_small, warp_mode, criteria, None, 5)
                wm = wm_small.copy()
                wm[0, 2] *= scale
                wm[1, 2] *= scale
                warp_matrix = wm
            else:
                _ = _cv2.findTransformECC(ref_n, mov_n, warp_matrix, warp_mode, criteria, None, 5)

            aligned = _cv2.warpAffine(
                mov,
                warp_matrix,
                (W, H),
                flags=_cv2.INTER_LINEAR + _cv2.WARP_INVERSE_MAP,
                borderMode=_cv2.BORDER_REPLICATE,
            ).astype(_np.float32)
            out[:, :, k] = aligned
        except Exception:
            # fallback: apply only the estimated translation or leave as-is
            try:
                aligned = _cv2.warpAffine(
                    mov,
                    warp_matrix,
                    (W, H),
                    flags=_cv2.INTER_LINEAR + _cv2.WARP_INVERSE_MAP,
                    borderMode=_cv2.BORDER_REPLICATE,
                ).astype(_np.float32)
                out[:, :, k] = aligned
            except Exception:
                out[:, :, k] = mov

    return out


def _roi_mediana_por_banda(img3d, titulo="Select a rectangle over the panel"):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector

    H, W, B = img3d.shape
    vis = img3d[:, :, :3] if B >= 3 else np.repeat(img3d[:, :, [0]], 3, axis=2)
    vmin, vmax = float(np.nanmin(vis)), float(np.nanmax(vis))
    vis_n = (vis - vmin) / (vmax - vmin + 1e-12)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(vis_n)
    ax.set_title(titulo + "\n(Drag with the mouse. ENTER confirms • ESC cancels)")
    ax.axis("off")

    rect_store = {"rect": None}

    def _on_select(ec, fc):
        # callback on mouse release (click-drag-release)
        if ec.xdata is None or fc.xdata is None:
            return
        x1, y1 = int(ec.xdata), int(ec.ydata)
        x2, y2 = int(fc.xdata), int(fc.ydata)
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(W, xmax)
        ymax = min(H, ymax)
        rect_store["rect"] = (xmin, xmax, ymin, ymax)

    def _on_key(event):
        if event.key == 'enter':
            plt.close(fig)
        elif event.key == 'escape':
            rect_store["rect"] = None
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', _on_key)

    face = (1.0, 1.0, 0.0, 0.25)  # yellow 25%
    edge = 'red'
    try:
        rs = RectangleSelector(
            ax,
            _on_select,
            useblit=False,
            button=[1],
            minspanx=3,
            minspany=3,
            spancoords='pixels',
            interactive=True,
            props=dict(facecolor=face, edgecolor=edge, linewidth=1.8),
        )
    except TypeError:
        rs = RectangleSelector(
            ax,
            _on_select,
            useblit=False,
            button=[1],
            minspanx=3,
            minspany=3,
            spancoords='pixels',
            interactive=True,
            rectprops=dict(facecolor=face, edgecolor=edge, linewidth=1.8),
        )

    plt.show()  # waits for ENTER/ESC

    if rect_store["rect"] is None:
        xmin, xmax, ymin, ymax = 0, W, 0, H
    else:
        xmin, xmax, ymin, ymax = rect_store["rect"]

    roi = img3d[ymin:ymax, xmin:xmax, :]
    med = np.nanmedian(roi.reshape(-1, B), axis=0)
    return med


def _ajustar_empirical_line(dn_targets_list, rho_list):
    """
    Band-by-band adjustment:
      - dn_targets_list: list of (B,) vectors with panel DN per band (one per panel)
      - rho_list: list of (B,) vectors OR scalars (one per panel). If scalar, it is
        broadcast to all bands.
    Returns:
      a, b (vectors of size B): reflectance = a * DN + b
    """
    dn_mat = np.stack(dn_targets_list, axis=0).astype(np.float32)  # (N, B)

    # Build rho_mat (N,B)
    rho_elems = []
    for r in rho_list:
        if np.isscalar(r):
            rho_elems.append(np.full(dn_mat.shape[1], float(r), dtype=np.float32))
        else:
            rr = np.asarray(r, dtype=np.float32)
            if rr.ndim == 0:
                rr = np.full(dn_mat.shape[1], float(rr), dtype=np.float32)
            rho_elems.append(rr)
    rho_mat = np.stack(rho_elems, axis=0)  # (N, B)

    N, B = dn_mat.shape
    a = np.zeros(B, dtype=np.float32)
    b = np.zeros(B, dtype=np.float32)
    for k in range(B):
        x = dn_mat[:, k]
        y = rho_mat[:, k]
        good = np.isfinite(x) & np.isfinite(y)
        xg, yg = x[good], y[good]
        if xg.size == 0:
            a[k], b[k] = 1.0, 0.0
        elif xg.size == 1:
            a[k], b[k] = (yg[0] / (xg[0] + 1e-12)), 0.0  # pure scaling
        else:
            X = np.column_stack([xg, np.ones_like(xg)])
            coef, *_ = np.linalg.lstsq(X, yg, rcond=None)
            a[k], b[k] = float(coef[0]), float(coef[1])
    return a, b


def _aplicar_empirical_line(cube_in, a, b, clip=True):
    out = cube_in.astype(np.float32, copy=True)
    for k in range(out.shape[2]):
        out[:, :, k] = a[k] * out[:, :, k] + b[k]
    if clip:
        np.clip(out, 0.0, 1.0, out=out)
    return out


def calibrar_cubo_por_paineis_com_bandas_gui(cube_in, wavelengths_in):
    paineis = _ui_escolher_paineis_com_rho()
    usar_rho_por_banda = _ui_pergunta_rho_por_banda()
    if usar_rho_por_banda is None:
        return cube_in, None
    if paineis is None:
        return cube_in, None
    if len(paineis) == 0:
        messagebox.showinfo("Panels", "No panel selected. Proceeding without correction.")
        return cube_in, None

    dn_targets = []
    rho_targets = []
    B = cube_in.shape[2]
    wls_cubo = np.array(wavelengths_in, dtype=float)

    for nome, rho in paineis:
        if usar_rho_por_banda:
            arqs, wls_painel, rho_bandas = _selecionar_arquivos_banda_para_painel(
                f"Panel {nome.upper()}",
                solicitar_rho_por_banda=True
            )
        else:
            arqs, wls_painel, _ = _selecionar_arquivos_banda_para_painel(
                f"Panel {nome.upper()}",
                solicitar_rho_por_banda=False,
                rho_geral=float(rho)
            )
            rho_bandas = [float(rho)] * len(wls_painel)
        if not arqs:
            continue
        stack = _carregar_painel_como_cubo(arqs)  # (H,W,Bp)
        wls_painel = np.array(wls_painel, dtype=float)

        if len(wls_painel) != B:
            messagebox.showerror(
                "Incompatible bands",
                f"Panel {nome.upper()} has {len(wls_painel)} bands, but the cube has {B} bands.\n"
                "For this workflow, select one file per band (same sensor) covering ALL bands."
            )
            return cube_in, None

        sort_idx = np.argsort(wls_painel)
        wls_painel = wls_painel[sort_idx]
        stack = stack[:, :, sort_idx]

        # *** Automatically align panel bands (robust) ***
        metodos = escolher_metodos_alinhamento()
        if not metodos:
            messagebox.showwarning("Warning", "Select at least one alignment method for the panel.")
            return cube_in, None
        params_por_metodo = coletar_parametros_para_metodos(metodos)
        stack, metodo_escolhido, _ = alinhar_stack_multimetodos(
            stack, idx_ref=0, metodos=metodos, params_por_metodo=params_por_metodo
        )

        # --- Show SSIM per method and allow manual choice (as in band alignment) ---
        try:
            _stack_pre = stack.copy() if hasattr(stack, "copy") else stack
            _tmp_alinhado, _tmp_melhor, _ssim_dict = alinhar_stack_multimetodos(
                _stack_pre, idx_ref=0, metodos=metodos, params_por_metodo=params_por_metodo
            )
            _resultados = {
                m: {"media": (None if _ssim_dict.get(m) is None else float(_ssim_dict.get(m)))}
                for m in metodos
            }
            try:
                _mostrar_resultados_ssim(_resultados)
            except Exception:
                pass
            _metodo_user = _escolher_metodo_prosseguir(list(metodos)) or metodo_escolhido
            if _metodo_user != metodo_escolhido:
                stack, metodo_escolhido, _ = alinhar_stack_multimetodos(
                    _stack_pre,
                    idx_ref=0,
                    metodos=[_metodo_user],
                    params_por_metodo=params_por_metodo,
                )
        except Exception:
            pass

        if np.max(np.abs(wls_painel - wls_cubo)) > 2.0:
            messagebox.showerror(
                "Different wavelengths",
                "Panel bands do not match cube bands (tolerance 2 nm). Adjust the wavelengths provided."
            )
            return cube_in, None

        med = _roi_mediana_por_banda(stack, f"Panel {nome.upper()}: select an area")
        if med.shape[0] != B:
            messagebox.showerror("Error", "Failed to extract median per band for the panel.")
            return cube_in, None

        dn_targets.append(med)
        rho_targets.append(np.array(rho_bandas, dtype=np.float32))

    if not dn_targets:
        messagebox.showwarning("Panels", "No valid panel was provided. Proceeding without correction.")
        return cube_in, None

    a, b = _ajustar_empirical_line(dn_targets, rho_targets)
    cube_out = _aplicar_empirical_line(cube_in, a, b, clip=True)

    detalhes = {
        "panels_used": [
            {"name": n, "reflectance": (r if not usar_rho_por_banda else "per_band")}
            for (n, r) in paineis
        ],
        "coefficients_a": a.tolist(),
        "coefficients_b": b.tolist(),
    }
    return cube_out, detalhes


def _ui_modo_calibracao_paineis():
    win = Toplevel()
    win.title("Panel/tile correction")
    win.geometry("420x220")
    win.grab_set()
    sel = tk.StringVar(value="por_banda")
    ttk.Label(win, text="How do you want to provide the reference?").pack(pady=(12, 6))
    ttk.Radiobutton(
        win,
        text="Panel files per band (traditional workflow)",
        variable=sel,
        value="por_banda",
    ).pack(anchor="w", padx=16)
    ttk.Radiobutton(
        win,
        text="Single panel/tile photo (1–3 tiles, with ROI)",
        variable=sel,
        value="foto_unica",
    ).pack(anchor="w", padx=16)
    ok = {"go": False}
    ttk.Button(
        win,
        text="Continue",
        command=lambda: (ok.update(go=True), win.destroy())
    ).pack(pady=10)
    win.wait_window()
    return sel.get() if ok["go"] else None


def alinhar_stack_multimetodos(stack_in, idx_ref=0, metodos=None, params_por_metodo=None):
    """Aligns a stack (H,W,B) with multiple methods and chooses the best by mean SSIM."""
    import numpy as _np
    H, W, B = stack_in.shape
    ref = stack_in[:, :, idx_ref].astype(_np.float32)
    if metodos is None or len(metodos) == 0:
        metodos = ["ecc"]
    params_por_metodo = params_por_metodo or {}
    ssim_dict = {}
    aligned_by_method = {}
    for metodo in metodos:
        params = params_por_metodo.get(metodo, {})
        alinhadas = []
        ssim_list = []
        for k in range(B):
            img = stack_in[:, :, k]
            if k == idx_ref:
                alinhadas.append(img)
                continue
            try:
                ali = alinhar_imagem(ref, img, metodo=metodo, params=params)
                if ali.shape != (H, W):
                    import cv2
                    ali = cv2.resize(ali, (W, H), interpolation=cv2.INTER_NEAREST)
                alinhadas.append(ali)
                try:
                    ssim_val = _ssim_norm(ref, ali)
                except Exception:
                    ssim_val = _np.nan
                ssim_list.append(ssim_val)
            except Exception:
                alinhadas.append(img)
                ssim_list.append(_np.nan)
        aligned = _np.stack(alinhadas, axis=-1).astype(_np.float32)
        aligned_by_method[metodo] = aligned
        ssim_arr = _np.array(ssim_list, dtype=_np.float32)
        ssim_dict[metodo] = float(_np.nanmean(ssim_arr)) if _np.isfinite(ssim_arr).any() else -_np.inf
    metodo_escolhido = max(ssim_dict, key=lambda m: ssim_dict[m])
    return aligned_by_method[metodo_escolhido], metodo_escolhido, ssim_dict


def calibrar_cubo_por_paineis_foto_unica_gui(cube_in, wavelengths_in):
    """Radiometric correction from a SINGLE panel/tile photo.
    Supports 1, 2 or 3 tiles with rectangular ROIs; applies Empirical Line.
    """
    from tkinter import filedialog as _fd, messagebox as _mb
    import numpy as _np
    try:
        from tifffile import imread as _tf_imread
    except Exception:
        _mb.showerror("Error", "tifffile not found to read the single panel photo.")
        return cube_in, None
    caminho = _fd.askopenfilename(
        title="Select the SINGLE panel/tile photo",
        filetypes=[
            ("Images", "*.tif *.tiff *.jpg *.jpeg *.png"),
            ("All files", "*.*"),
        ],
    )
    if not caminho:
        return cube_in, None
    img = _tf_imread(caminho)
    if img.ndim == 2:
        img = img[:, :, None]
    elif img.ndim != 3:
        _mb.showerror("Error", "Unsupported image format for the single panel photo.")
        return cube_in, None
    img = img.astype(_np.float32)
    H, W, C = img.shape
    B = cube_in.shape[2]

    win = Toplevel()
    win.title("Tiles")
    win.geometry("460x320")
    win.grab_set()
    ttk.Label(
        win,
        text="Select how many tiles and provide their nominal reflectance (0–1)"
    ).pack(pady=(10, 6))
    n_var = tk.IntVar(value=1)
    ttk.Radiobutton(win, text="1 tile", variable=n_var, value=1).pack(anchor="w", padx=16)
    ttk.Radiobutton(win, text="2 tiles", variable=n_var, value=2).pack(anchor="w", padx=16)
    ttk.Radiobutton(win, text="3 tiles", variable=n_var, value=3).pack(anchor="w", padx=16)
    entries = []
    tipos = ["White", "Gray", "Black"]
    vals_default = [0.99, 0.18, 0.03]
    frm = ttk.Frame(win)
    frm.pack(fill="x", padx=12, pady=8)
    for i in range(3):
        f = ttk.Frame(frm)
        f.pack(fill="x", pady=3)
        ttk.Label(f, text=f"Tile {i + 1} (e.g. {tipos[i]}):").pack(side="left")
        e = tk.DoubleVar(value=vals_default[i])
        ttk.Entry(f, textvariable=e, width=8).pack(side="left", padx=8)
        entries.append(e)
    ok = {"go": False}
    ttk.Button(win, text="Continue", command=lambda: (ok.update(go=True), win.destroy())).pack(pady=10)
    win.wait_window()
    if not ok["go"]:
        return cube_in, None
    n_p = n_var.get()
    rhos = [entries[i].get() for i in range(n_p)]

    dn_list = []
    for i in range(n_p):
        med = _roi_mediana_por_banda(
            img,
            titulo=f"Select ROI for Tile {i + 1} and press ENTER (ESC to confirm)",
        )
        dn_list.append(med)   # shape (C,)
    dn_mat = _np.stack(dn_list, axis=0)  # (n_p, C)
    rho_vec = _np.array(rhos, dtype=_np.float32)

    def fit_ab(x, r):
        x = _np.asarray(x, dtype=_np.float32)
        r = _np.asarray(r, dtype=_np.float32)
        good = _np.isfinite(x) & _np.isfinite(r)
        xg, rg = x[good], r[good]
        if xg.size == 0:
            return 1.0, 0.0
        if xg.size == 1:
            return float(rg[0] / (xg[0] + 1e-12)), 0.0
        if xg.size == 2:
            a = float((rg[1] - rg[0]) / ((xg[1] - xg[0]) + 1e-12))
            b = float(rg[0] - a * xg[0])
            return a, b
        A = _np.column_stack([xg, _np.ones_like(xg)])
        coef, *_ = _np.linalg.lstsq(A, rg, rcond=None)
        return float(coef[0]), float(coef[1])

    if C == B:
        a = _np.zeros(B, _np.float32)
        b = _np.zeros(B, _np.float32)
        for k in range(B):
            ak, bk = fit_ab(dn_mat[:, k], rho_vec)
            a[k], b[k] = ak, bk
    else:
        x_mean = _np.nanmean(dn_mat, axis=1)
        a1, b1 = fit_ab(x_mean, rho_vec)
        a = _np.full(B, a1, _np.float32)
        b = _np.full(B, b1, _np.float32)

    cube = cube_in.astype(_np.float32)
    cube_corr = np.empty_like(cube)
    for k in range(B):
        cube_corr[:, :, k] = a[k] * cube[:, :, k] + b[k]
    cube_corr = np.clip(cube_corr, 0.0, 1.0)

    detalhes = {
        "mode": "single_photo",
        "reference_file": caminho,
        "n_tiles": int(n_p),
        "reflectances": [float(r) for r in rhos],
        "coefficients_a": a.tolist(),
        "coefficients_b": b.tolist(),
    }
    return cube_corr, detalhes


def _try_corr_paineis(func, cube_in, waves):
    """Executes the panel correction function and guarantees a (cube, details) return.
    If the function returns None or only the cube, normalize to (cube, None).
    """
    res = func(cube_in, waves)
    if res is None:
        return cube_in, None
    try:
        cube_out, detalhes = res
        return cube_out, detalhes
    except Exception:
        # If only the cube was returned, wrap it with detalhes=None
        return res, None


# ============================
# NEW ALIGNMENT FLOW: AUTO vs MANUAL + PARAMETERS
# ============================
try:
    import tkinter as _tk
    from tkinter import ttk as _ttk
except Exception:
    _tk = None
    _ttk = None

_ULTIMA_PARAMS_POR_METODO = None  # cache when user chooses manual mode


def _ui_escolher_modo_alinhamento():
    """
    Opens a window asking:
    - Test several methods (automatic)
    - Choose a specific method (manual, with parameters)
    Returns 'auto' or 'manual'. If the user closes, returns 'auto' by default.
    """
    if _tk is None:
        return "auto"
    win = _tk.Toplevel()
    win.title("Alignment mode")
    win.geometry("420x180")
    win.grab_set()
    sel = _tk.StringVar(value="auto")
    _tk.Label(win, text="How do you want to proceed with alignment?").pack(pady=(12, 6))
    _ttk.Radiobutton(
        win,
        text="Test several methods (automatic)",
        variable=sel,
        value="auto",
    ).pack(anchor="w", padx=16)
    _ttk.Radiobutton(
        win,
        text="Choose a specific method (manual, with parameters)",
        variable=sel,
        value="manual",
    ).pack(anchor="w", padx=16)
    ok = {"go": False}
    _ttk.Button(
        win,
        text="Continue",
        command=lambda: (ok.update(go=True), win.destroy())
    ).pack(pady=10)
    win.wait_window()
    return sel.get() if ok["go"] else "auto"


def _ui_escolher_um_metodo():
    """Simple window to choose ONE method."""
    metodos = ["orb", "sift", "akaze", "ecc", "template", "mi", "superglue"]
    nomes = {
        "orb": "ORB keypoints",
        "sift": "SIFT keypoints",
        "akaze": "AKAZE keypoints",
        "ecc": "ECC (intensity-based)",
        "template": "Template matching",
        "mi": "Mutual information",
        "superglue": "SuperGlue (deep matching)",
    }
    if _tk is None:
        return "ecc"
    win = _tk.Toplevel()
    win.title("Choose alignment method")
    win.geometry("380x180")
    win.grab_set()
    _tk.Label(win, text="Select the method:").pack(pady=8)
    opcoes = [f"{m} — {nomes[m]}" for m in metodos]
    var = _tk.StringVar(value=opcoes[0])
    combo = _ttk.Combobox(win, values=opcoes, textvariable=var, state="readonly")
    combo.current(0)
    combo.pack(pady=6, fill="x", padx=12)
    out = {"m": None}

    def confirmar():
        s = combo.get().split(" — ")[0].strip()
        out["m"] = s
        win.destroy()

    _tk.Button(win, text="Confirm", command=confirmar).pack(pady=8)
    win.wait_window()
    return out["m"]


def _ui_parametros_generico(title, campos):
    """
    Creates a parameter window based on a dict:
    {name: (default, kind, options)}.
    kind: 'float' | 'int' | 'str' | 'combo'
    options: list of strings for combo
    Returns a dict with converted values.
    """
    if _tk is None:
        # headless fallback
        return {k: v[0] for k, v in campos.items()}
    win = _tk.Toplevel()
    win.title(title)
    win.geometry("440x520")
    win.grab_set()
    frm = _tk.Frame(win)
    frm.pack(fill="both", expand=True, padx=12, pady=8)
    vars_map = {}
    for i, (nome, (default, kind, *rest)) in enumerate(campos.items()):
        row = _tk.Frame(frm)
        row.pack(fill="x", pady=4)
        _tk.Label(row, text=nome).pack(side="left")
        if kind == "combo":
            opts = rest[0] if rest else []
            var = _tk.StringVar(value=str(default))
            cb = _ttk.Combobox(row, values=opts, textvariable=var, state="readonly", width=24)
            if default in opts:
                cb.set(default)
            cb.pack(side="right")
            vars_map[nome] = (var, kind)
        else:
            var = _tk.StringVar(value=str(default))
            ent = _tk.Entry(row, textvariable=var, width=26)
            ent.pack(side="right")
            vars_map[nome] = (var, kind)
    out = {"ok": False}

    def _ok():
        out["ok"] = True
        win.destroy()

    _tk.Button(win, text="Confirm", command=_ok).pack(pady=10)
    win.wait_window()
    if not out["ok"]:
        return {k: v[0] for k, v in campos.items()}
    # convert
    res = {}
    for nome, (var, kind) in vars_map.items():
        val = var.get()
        try:
            if kind == "int":
                res[nome] = int(val)
            elif kind == "float":
                res[nome] = float(val)
            else:
                res[nome] = val
        except Exception:
            res[nome] = campos[nome][0]
    return res


def pedir_parametros_orb():
    campos = {
        "nfeatures": (5000, "int"),
        "scaleFactor": (1.2, "float"),
        "nlevels": (8, "int"),
        "edgeThreshold": (31, "int"),
        "patchSize": (31, "int"),
        "good_match_percent": (0.15, "float"),
        "ransac_thresh": (3.0, "float"),
        "warp_interp": ("nearest", "combo", ["nearest", "bilinear", "bicubic"]),
        "border_mode": ("replicate", "combo", ["constant", "replicate", "reflect", "wrap"]),
    }
    return _ui_parametros_generico("Parameters — ORB", campos)


def pedir_parametros_sift():
    campos = {
        "contrastThreshold": (0.04, "float"),
        "edgeThreshold": (10, "int"),
        "nOctaveLayers": (3, "int"),
        "sigma": (1.6, "float"),
        "ratio_test": (0.75, "float"),
        "ransac_thresh": (3.0, "float"),
        "warp_interp": ("nearest", "combo", ["nearest", "bilinear", "bicubic"]),
        "border_mode": ("replicate", "combo", ["constant", "replicate", "reflect", "wrap"]),
    }
    return _ui_parametros_generico("Parameters — SIFT", campos)


def pedir_parametros_akaze():
    campos = {
        "threshold": (0.001, "float"),
        "descriptor_size": (0, "int"),
        "nOctaves": (4, "int"),
        "nOctaveLayers": (4, "int"),
        "ransac_thresh": (3.0, "float"),
        "good_match_percent": (0.20, "float"),
        "warp_interp": ("nearest", "combo", ["nearest", "bilinear", "bicubic"]),
        "border_mode": ("replicate", "combo", ["constant", "replicate", "reflect", "wrap"]),
    }
    return _ui_parametros_generico("Parameters — AKAZE", campos)


def pedir_parametros_ecc():
    campos = {
        "warp_mode": ("affine", "combo", ["translation", "euclidean", "affine", "homography"]),
        "number_of_iterations": (100, "int"),
        "termination_eps": (1e-6, "float"),
        "gaussFiltSize": (0, "int"),
        "warp_interp": ("nearest", "combo", ["nearest", "bilinear", "bicubic"]),
        "border_mode": ("replicate", "combo", ["constant", "replicate", "reflect", "wrap"]),
    }
    return _ui_parametros_generico("Parameters — ECC", campos)


def pedir_parametros_template():
    campos = {
        "method": ("TM_CCOEFF_NORMED", "combo", ["TM_CCOEFF_NORMED", "TM_CCORR_NORMED", "TM_SQDIFF_NORMED"]),
        "resize_factor": (1.0, "float"),
        "warp_interp": ("nearest", "combo", ["nearest", "bilinear", "bicubic"]),
        "border_mode": ("replicate", "combo", ["constant", "replicate", "reflect", "wrap"]),
    }
    return _ui_parametros_generico("Parameters — Template Matching", campos)


def pedir_parametros_mi():
    # Uses template as proxy for now
    return pedir_parametros_template()


def pedir_parametros_superglue():
    campos = {
        "match_threshold": (0.2, "float"),
        "sinkhorn_iterations": (20, "int"),
        "max_keypoints": (2048, "int"),
    }
    return _ui_parametros_generico("Parameters — SuperGlue", campos)


def escolher_metodos_alinhamento():
    """
    NEW VERSION:
    1) Asks mode: auto (multi-methods) or manual (one method + parameters)
    2) If auto: returns list of selected methods (checkboxes),
       _ULTIMA_PARAMS_POR_METODO = None
    3) If manual: returns [chosen_method] and stores _ULTIMA_PARAMS_POR_METODO = {method: params}
    """
    global _ULTIMA_PARAMS_POR_METODO
    modo = _ui_escolher_modo_alinhamento()
    metodos = ["orb", "sift", "akaze", "ecc", "template", "mi", "superglue"]
    nomes = {
        "orb": "ORB keypoints",
        "sift": "SIFT keypoints",
        "akaze": "AKAZE keypoints",
        "ecc": "ECC (intensity-based)",
        "template": "Template matching",
        "mi": "Mutual information",
        "superglue": "SuperGlue (deep matching)",
    }
    if modo == "manual":
        m = _ui_escolher_um_metodo()
        f = {
            "orb": pedir_parametros_orb,
            "sift": pedir_parametros_sift,
            "akaze": pedir_parametros_akaze,
            "ecc": pedir_parametros_ecc,
            "template": pedir_parametros_template,
            "mi": pedir_parametros_mi,
            "superglue": pedir_parametros_superglue,
        }.get(m)
        params = f() if callable(f) else {}
        _ULTIMA_PARAMS_POR_METODO = {m: params}
        return [m]

    # AUTO mode: UI with checkboxes
    _ULTIMA_PARAMS_POR_METODO = None
    if _tk is None:
        return ["orb", "sift", "akaze", "ecc", "template", "mi", "superglue"]
    win = _tk.Toplevel()
    win.title("Select alignment methods")
    win.geometry("360x380")
    win.grab_set()
    _tk.Label(win, text="Select the methods you want to apply:").pack(pady=8)
    vars_map = {}
    frm = _tk.Frame(win)
    frm.pack(fill="both", expand=True, padx=12, pady=6)
    for m in metodos:
        v = _tk.IntVar(value=1 if m in ("ecc", "orb", "sift") else 0)
        _tk.Checkbutton(frm, text=f"{m} — {nomes[m]}", variable=v).pack(anchor="w")
        vars_map[m] = v
    out = {"ok": False}

    def _ok():
        out["ok"] = True
        win.destroy()

    _tk.Button(win, text="Confirm", command=_ok).pack(pady=10)
    win.wait_window()
    sel = [m for m, v in vars_map.items() if v.get() == 1] if out["ok"] else []
    if not sel:
        # ensure at least one
        sel = ["ecc"]
    return sel


def coletar_parametros_para_metodos(metodos):
    """
    NEW VERSION:
    - If the user chose MANUAL, reuse _ULTIMA_PARAMS_POR_METODO.
    - If AUTO, optionally ask for parameters per method (dialogs in sequence).
      To use defaults, simply return {} per method.
    """
    global _ULTIMA_PARAMS_POR_METODO
    if _ULTIMA_PARAMS_POR_METODO:
        return {m: _ULTIMA_PARAMS_POR_METODO.get(m, {}) for m in metodos}

    if _tk is None:
        return {m: {} for m in metodos}
    resp = _tk.messagebox.askyesno("Parameters", "Do you want to adjust parameters of the selected methods?")
    if not resp:
        return {m: {} for m in metodos}

    name2func = {
        "superglue": pedir_parametros_superglue,
        "orb": pedir_parametros_orb,
        "sift": pedir_parametros_sift,
        "ecc": pedir_parametros_ecc,
        "template": pedir_parametros_template,
        "akaze": pedir_parametros_akaze,
        "mi": pedir_parametros_mi,
    }
    params_por_metodo = {}
    for m in metodos:
        f = name2func.get(m)
        params_por_metodo[m] = f() if callable(f) else {}
    return params_por_metodo


def selecionar_bandas_gui():
    """
    Window to select a subset of bands from the current cube.
    Requires cube/wavelengths to be defined. Updates global state with the subcube.
    """
    import tkinter as _tk
    from tkinter import messagebox as _mb

    try:
        cube = get_cube()
        wavelengths = get_wavelengths()
    except Exception:
        cube = None
        wavelengths = None

    if cube is None:
        _mb.showwarning("No cube", "No cube loaded. Import a .npy or align bands first.")
        return

    if cube.ndim != 3:
        _mb.showerror("Invalid format", f"Expected cube (H, W, B). Got shape={getattr(cube, 'shape', None)}")
        return

    H, W, B = cube.shape
    labels = []
    if wavelengths is not None and len(wavelengths) == B:
        try:
            wls = [float(x) for x in (wavelengths.tolist() if hasattr(wavelengths, "tolist") else list(wavelengths))]
            labels = [f"{i:03d} — {wls[i]:.2f} nm" for i in range(B)]
        except Exception:
            labels = [f"{i:03d} — band" for i in range(B)]
    else:
        labels = [f"{i:03d} — band" for i in range(B)]

    win = _tk.Toplevel()
    win.title("Select specific bands")
    win.geometry("460x520")
    win.grab_set()

    _tk.Label(win, text=f"Select the bands you want to keep (total: {B})").pack(pady=(10, 6))

    frame_list = _tk.Frame(win)
    frame_list.pack(fill="both", expand=True, padx=12)
    sb = _tk.Scrollbar(frame_list)
    sb.pack(side="right", fill="y")

    lb = _tk.Listbox(frame_list, selectmode=_tk.EXTENDED, yscrollcommand=sb.set)
    for lab in labels:
        lb.insert(_tk.END, lab)
    lb.pack(fill="both", expand=True)
    sb.config(command=lb.yview)

    _tk.Label(
        win,
        text="Tip: use Ctrl/Cmd for multi-selection, Shift for ranges."
    ).pack(pady=(6, 6))

    def confirmar():
        sel = list(lb.curselection())
        if not sel:
            _mb.showwarning("Nothing selected", "Select at least one band.")
            return
        sel_sorted = sorted(sel)
        import numpy as _np
        try:
            novo_cubo = cube[:, :, sel_sorted]
        except Exception as e:
            _mb.showerror("Filter error", f"Could not create subcube:\n{e}")
            return

        novas_wls = None
        if wavelengths is not None and len(wavelengths) == B:
            try:
                wlist = wavelengths.tolist() if hasattr(wavelengths, "tolist") else list(wavelengths)
                novas_wls = [wlist[i] for i in sel_sorted]
            except Exception:
                novas_wls = None

        set_cube(novo_cubo)
        if novas_wls is not None:
            set_wavelengths(_np.array(novas_wls, dtype=float))

        _mb.showinfo("Bands updated", f"Cube now has {novo_cubo.shape[2]} bands.")
        win.destroy()

    btns = _tk.Frame(win)
    btns.pack(pady=10)
    _tk.Button(btns, text="Keep selected", command=confirmar).pack(side="left", padx=8)
    _tk.Button(btns, text="Cancel", command=win.destroy).pack(side="left", padx=8)


# =====================================================
# OVERRIDES — Clear separation of import vs panel flows
# =====================================================

def cadastrar_bandas_gui():
    """
    EXCLUSIVE window to import .tif bands WITHOUT georeference.
    Fields: file, nm. (NO rho here)
    Returns True if the user clicked 'Save and use'; otherwise False.
    """
    import os, json, re
    import tkinter as _tk
    from tkinter import filedialog as _fd, messagebox as _mb, simpledialog as _sd, ttk as _ttk

    global _cadastro_atual

    win = _tk.Toplevel()
    win.title("Band registration (.tif without georeference)")
    win.geometry("860x540")
    win.grab_set()

    resultado = {"ok": False}

    frame = _tk.Frame(win)
    frame.pack(fill="both", expand=True, padx=14, pady=(12, 8))
    cols = ("arquivo", "nm")
    tv = _ttk.Treeview(frame, columns=cols, show="headings", selectmode="extended")
    tv.heading("arquivo", text="File (.tif)")
    tv.heading("nm", text="Wavelength (nm)")
    tv.column("arquivo", width=600, anchor="w")
    tv.column("nm", width=160, anchor="center")

    sb = _tk.Scrollbar(frame, orient="vertical", command=tv.yview)
    tv.configure(yscrollcommand=sb.set)
    tv.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    def _parse_nm_from_name(name):
        m = re.search(r'(\d{3,4})\s*nm', name, re.I)
        if m:
            return int(m.group(1))
        m = re.search(r'[_\-](\d{3,4})[_\-. ]', name, re.I)
        if m:
            return int(m.group(1))
        return None

    def _adicionar():
        paths = _fd.askopenfilenames(
            title="Select .tif files (one band per file)",
            filetypes=[("TIFF", "*.tif *.tiff")],
        )
        if not paths:
            return
        for p in paths:
            nm_guess = _parse_nm_from_name(os.path.basename(p))
            if nm_guess is None:
                try:
                    nm_guess = _sd.askinteger(
                        "Identify band",
                        f"What is the wavelength (nm) for:\n{os.path.basename(p)}"
                    )
                except Exception:
                    nm_guess = None
            tv.insert("", "end", values=(p, "" if nm_guess is None else int(nm_guess)))

    def _editar_nm():
        sel = tv.selection()
        if not sel:
            _mb.showwarning("No selection", "Select a row to edit nm.")
            return
        for iid in sel:
            path_atual, nm_atual = tv.item(iid, "values")
            try:
                atual = None if nm_atual in ("", None) else int(nm_atual)
            except Exception:
                atual = None
            novo_nm = _sd.askinteger(
                "Edit wavelength",
                "Wavelength (nm)",
                initialvalue=atual
            )
            if novo_nm is None:
                continue
            tv.item(iid, values=(path_atual, int(novo_nm)))

    def _remover():
        sel = tv.selection()
        if not sel:
            _mb.showwarning("No selection", "Select at least one row to remove.")
            return
        for iid in sel:
            tv.delete(iid)

    def _limpar():
        for iid in tv.get_children():
            tv.delete(iid)

    def _ordenar_por_nm():
        itens = [(iid, tv.item(iid, "values")) for iid in tv.get_children()]

        def nm_val(v):
            try:
                return int(v[1])
            except Exception:
                return 10**9

        itens.sort(key=lambda x: nm_val(x[1]))
        for iid, _ in itens:
            tv.move(iid, "", "end")

    def _salvar_e_usar():
        global _cadastro_atual
        itens = [tv.item(iid, "values") for iid in tv.get_children()]
        if not itens:
            _mb.showwarning("Empty", "Add at least one band.")
            return
        arquivos, bandas_nm = [], []
        for (p, nm) in itens:
            if not p or not os.path.exists(p):
                _mb.showerror("Invalid file", f"File not found:\n{p}")
                return
            try:
                nm_int = int(nm)
            except Exception:
                _mb.showerror("Invalid value", f"Invalid wavelength:\n{nm}")
                return
            arquivos.append(p)
            bandas_nm.append(nm_int)

        pares = sorted(zip(bandas_nm, arquivos), key=lambda x: x[0])
        bandas_nm = [p[0] for p in pares]
        arquivos = [p[1] for p in pares]

        _cadastro_atual = {"arquivos": arquivos, "bandas_nm": bandas_nm}

        diretorio = os.path.dirname(arquivos[0]) if arquivos else os.getcwd()
        destino = os.path.join(diretorio, "cadastro_bandas.json")
        try:
            with open(destino, "w", encoding="utf-8") as f:
                json.dump(_cadastro_atual, f, indent=4, ensure_ascii=False)
            _cadastro_atual["path_json"] = destino
            _mb.showinfo("Registration saved", f"Registration saved at:\n{destino}")
            resultado["ok"] = True
            win.destroy()
        except Exception as e:
            _mb.showerror("Save error", f"Could not save registration (.json):\n{e}")

    btns = _tk.Frame(win)
    btns.pack(fill="x", padx=14, pady=(4, 12))
    _tk.Button(btns, text="Add bands", command=_adicionar).pack(side="left", padx=4)
    _tk.Button(btns, text="Edit nm", command=_editar_nm).pack(side="left", padx=4)
    _tk.Button(btns, text="Remove", command=_remover).pack(side="left", padx=4)
    _tk.Button(btns, text="Clear", command=_limpar).pack(side="left", padx=4)
    _tk.Button(btns, text="Sort by nm", command=_ordenar_por_nm).pack(side="left", padx=12)
    _tk.Button(btns, text="Save and use", command=_salvar_e_usar).pack(side="right", padx=4)
    _tk.Button(btns, text="Cancel", command=win.destroy).pack(side="right", padx=4)

    win.wait_window()
    return resultado["ok"]


def cadastrar_paineis_gui(titulo="Select panels (per band)"):
    """
    EXCLUSIVE window to import PANEL IMAGES and capture per-band metadata.
    Fields: file, nm, rho (0–1).
    Returns (arquivos, bandas_nm, rho_bandas) or ([], [], []) if cancelled.
    """
    import os, json, re
    import tkinter as _tk
    from tkinter import filedialog as _fd, messagebox as _mb, simpledialog as _sd, ttk as _ttk

    win = _tk.Toplevel()
    win.title(titulo)
    win.geometry("900x560")
    win.grab_set()

    retorno = {"val": ([], [], [])}

    frame = _tk.Frame(win)
    frame.pack(fill="both", expand=True, padx=14, pady=(12, 8))
    cols = ("arquivo", "nm", "rho")
    tv = _ttk.Treeview(frame, columns=cols, show="headings", selectmode="extended")
    tv.heading("arquivo", text="File (.tif)")
    tv.heading("nm", text="Wavelength (nm)")
    tv.heading("rho", text="Panel reflectance (0–1)")
    tv.column("arquivo", width=560, anchor="w")
    tv.column("nm", width=140, anchor="center")
    tv.column("rho", width=160, anchor="center")

    sb = _tk.Scrollbar(frame, orient="vertical", command=tv.yview)
    tv.configure(yscrollcommand=sb.set)
    tv.pack(side="left", fill="both", expand=True)
    sb.pack(side="right", fill="y")

    def _parse_nm_from_name(name):
        m = re.search(r'(\d{3,4})\s*nm', name, re.I)
        if m:
            return int(m.group(1))
        m = re.search(r'[_\-](\d{3,4})[_\-. ]', name, re.I)
        if m:
            return int(m.group(1))
        return None

    def _adicionar():
        paths = _fd.askopenfilenames(
            title="Select .tif files (panels; one band per file)",
            filetypes=[("TIFF", "*.tif *.tiff")],
        )
        if not paths:
            return
        for p in paths:
            nm_guess = _parse_nm_from_name(os.path.basename(p))
            if nm_guess is None:
                try:
                    nm_guess = _sd.askinteger(
                        "Identify band",
                        f"What is the wavelength (nm) for:\n{os.path.basename(p)}"
                    )
                except Exception:
                    nm_guess = None
            tv.insert("", "end", values=(p, "" if nm_guess is None else int(nm_guess), ""))

    def _editar_nm():
        sel = tv.selection()
        if not sel:
            _mb.showwarning("No selection", "Select a row to edit nm.")
            return
        for iid in sel:
            vals = list(tv.item(iid, "values"))
            while len(vals) < 3:
                vals.append("")
            try:
                atual = None if vals[1] in ("", None) else int(vals[1])
            except Exception:
                atual = None
            novo_nm = _sd.askinteger(
                "Edit wavelength",
                "Wavelength (nm)",
                initialvalue=atual
            )
            if novo_nm is None:
                continue
            vals[1] = int(novo_nm)
            tv.item(iid, values=tuple(vals))

    def _editar_rho():
        sel = tv.selection()
        if not sel:
            _mb.showwarning("No selection", "Select a row to edit reflectance (0–1).")
            return
        for iid in sel:
            vals = list(tv.item(iid, "values"))
            while len(vals) < 3:
                vals.append("")
            try:
                atual = None if vals[2] in ("", None) else float(vals[2])
            except Exception:
                atual = None
            rho = _sd.askfloat(
                "Panel reflectance",
                "Reflectance (0–1)",
                minvalue=0.0,
                maxvalue=1.0,
                initialvalue=atual
            )
            if rho is None:
                continue
            vals[2] = f"{float(max(0.0, min(1.0, rho))):.4f}"
            tv.item(iid, values=tuple(vals))

    def _remover():
        sel = tv.selection()
        if not sel:
            _mb.showwarning("No selection", "Select at least one row to remove.")
            return
        for iid in sel:
            tv.delete(iid)

    def _limpar():
        for iid in tv.get_children():
            tv.delete(iid)

    def _ordenar_por_nm():
        itens = [(iid, tv.item(iid, "values")) for iid in tv.get_children()]

        def nm_val(v):
            try:
                return int(v[1])
            except Exception:
                return 10**9

        itens.sort(key=lambda x: nm_val(x[1]))
        for iid, _ in itens:
            tv.move(iid, "", "end")

    def _salvar():
        itens = [tv.item(iid, "values") for iid in tv.get_children()]
        if not itens:
            _mb.showwarning("Empty", "Add at least one band.")
            return
        arquivos, bandas_nm, rho_bandas = [], [], []
        for vals in itens:
            vals = list(vals)
            while len(vals) < 3:
                vals.append("")
            p, nm, rho = vals[0], vals[1], vals[2]
            if not p or not os.path.exists(p):
                _mb.showerror("Invalid file", f"File not found:\n{p}")
                return
            try:
                nm_int = int(nm)
            except Exception:
                _mb.showerror("Invalid value", f"Invalid wavelength:\n{nm}")
                return
            arquivos.append(p)
            bandas_nm.append(nm_int)
            try:
                if rho not in ("", None):
                    v = float(rho)
                    v = max(0.0, min(1.0, v))
                    rho_bandas.append(v)
                else:
                    rho_bandas.append(None)
            except Exception:
                rho_bandas.append(None)
        pares = sorted(zip(bandas_nm, arquivos, rho_bandas), key=lambda x: x[0])
        bandas_nm = [p[0] for p in pares]
        arquivos = [p[1] for p in pares]
        rho_bandas = [p[2] for p in pares]
        retorno["val"] = (arquivos, bandas_nm, rho_bandas)
        win.destroy()

    btns = _tk.Frame(win)
    btns.pack(fill="x", padx=14, pady=(4, 12))
    _tk.Button(btns, text="Add panels", command=_adicionar).pack(side="left", padx=4)
    _tk.Button(btns, text="Edit nm", command=_editar_nm).pack(side="left", padx=4)
    _tk.Button(btns, text="Edit rho", command=_editar_rho).pack(side="left", padx=4)
    _tk.Button(btns, text="Remove", command=_remover).pack(side="left", padx=4)
    _tk.Button(btns, text="Clear", command=_limpar).pack(side="left", padx=4)
    _tk.Button(btns, text="Sort by nm", command=_ordenar_por_nm).pack(side="left", padx=12)
    _tk.Button(btns, text="Use for correction", command=_salvar).pack(side="right", padx=4)
    _tk.Button(btns, text="Cancel", command=win.destroy).pack(side="right", padx=4)

    win.wait_window()
    return retorno["val"]


def _selecionar_arquivos_banda_para_painel(titulo, solicitar_rho_por_banda=False, rho_geral=None):
    """
    Now ALWAYS uses the exclusive PANEL window.
    Returns (arquivos, bandas_nm, rho_bandas).
    - If solicitar_rho_por_banda=True and user did not fill in, asks after the window (window already closed).
    - If rho_geral is provided, fills automatically.
    """
    from tkinter import simpledialog, messagebox
    arquivos, bandas_nm, rho_bandas = cadastrar_paineis_gui(titulo)

    if not arquivos:
        return [], [], []

    if solicitar_rho_por_banda:
        if (not rho_bandas) or all(v is None for v in rho_bandas):
            rho_bandas = []
            for p, nm in zip(arquivos, bandas_nm):
                try:
                    rho = simpledialog.askfloat(
                        "Panel reflectance",
                        f"Reflectance (0–1) for band {int(nm)} nm\n{os.path.basename(p)}",
                        minvalue=0.0,
                        maxvalue=1.0,
                    )
                except Exception:
                    rho = None
                if rho is None:
                    messagebox.showwarning(
                        "Panels",
                        "Operation cancelled while providing band reflectance."
                    )
                    return [], [], []
                rho_bandas.append(float(max(0.0, min(1.0, rho))))
    elif rho_geral is not None:
        v = float(max(0.0, min(1.0, float(rho_geral))))
        rho_bandas = [v] * len(arquivos)

    return arquivos, bandas_nm, rho_bandas

