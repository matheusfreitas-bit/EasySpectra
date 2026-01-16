# EasySpectra – Installation Guide

This document explains how to install and run **EasySpectra** on a local machine
for research and commercial use.

---

## 1. System requirements

- Operating system:
  - Windows 10 or later
  - macOS 12 or later
- Python:
  - Python 3.10 or newer (3.11 recommended)
- Hardware:
  - At least 8 GB RAM (16 GB recommended)
  - Enough disk space for hyperspectral cubes and orthomosaics

---

## 2. Project structure

Your project folder should be named:
EasySpectra_EN/

and contain at least:
EasySpectra_EN/
 ├─ src/
 │   └─ easyspectra/
 │       ├─ __init__.py
 │       ├─ interface.py
 │       ├─ funcoes_importacao.py
 │       ├─ preprocessamento.py
 │       ├─ analise_espectral.py
 │       ├─ normalizacao.py
 │       ├─ metodos_alinhamento.py
 │       ├─ models/
 │       │   ├─ __init__.py
 │       │   ├─ matching.py
 │       │   ├─ superpoint.py
 │       │   ├─ superglue.py
 │       │   ├─ metodos_superglue.py
 │       │   └─ ...
 │       └─ ...
 ├─ docs/
 │   └─ INSTALL_en.md
 ├─ requirements.txt
 └─ README.md

---

## 3. Creating and activating a virtual environment

Open a terminal and go to the project root:
cd EasySpectra_EN

3.1. Create the virtual environment:
- macOS/Linux:
  python3 -m venv .venv
- Windows:
  python -m venv .venv

3.2. Activate the virtual environment:
- macOS/Linux:
  source .venv/bin/activate
- Windows:
  .venv\Scripts\activate

When active, your terminal will show (.venv).

---

## 4. Installing dependencies

Dependencies are listed in:
requirements.txt

With the environment active, install them:
pip install --upgrade pip
pip install -r requirements.txt

Typical dependencies:
numpy, matplotlib, scikit-image, opencv-python, rasterio, torch,
Pillow, tifffile, tkinter.

Some packages (like rasterio) may require GDAL or system libraries.

---

## 5. Running EasySpectra

The application is in:
src/easyspectra

Run the graphical interface:
cd EasySpectra_EN/src
python -m easyspectra.interface

---

## 6. Quick installation checklist

1. Download or clone EasySpectra_EN.
2. Open terminal → cd EasySpectra_EN
3. Create and activate environment:
   python -m venv .venv
   source .venv/bin/activate  (macOS/Linux)
   .venv\Scripts\activate   (Windows)
4. Install dependencies:
   pip install -r requirements.txt
5. Go to src:
   cd src
6. Run:
   python -m easyspectra.interface

---

## 7. Troubleshooting

7.1. ModuleNotFoundError: No module named 'easyspectra'
- Ensure you are running inside src:
  cd EasySpectra_EN/src
  python -m easyspectra.interface
- Check that easyspectra/ contains __init__.py.

7.2. Problems installing rasterio, GDAL, OpenCV:
- macOS: brew install gdal
- Linux: use apt, yum, or dnf
- Windows: consider Anaconda or precompiled wheels

7.3. Tkinter not found:
- Windows: reinstall Python with Tcl/Tk
- Linux: sudo apt install python3-tk

If issues persist, provide OS, Python version, and the full error message.
