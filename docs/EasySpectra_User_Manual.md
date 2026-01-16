# EasySpectra – User Manual

This document provides a practical guide on how to use the EasySpectra application.

---

## 1. Launching the Application

After installation, navigate to:

cd EasySpectra_EN/src

Then start the interface:

python -m easyspectra.interface

The main window will appear with tabs for:
- Preprocessing
- Spectral Analysis
- RGB Exploration
- GeoImport

---

## 2. Importing a Spectral Cube

In the Preprocessing tab:
1. Click "Load Cube"
2. Choose a hyperspectral image file (TIFF, NPY)
3. The cube will load into memory

Supported formats:
- .tiff and .tif
- .npy
- ODM-generated orthomosaics (GeoImport)

---

## 3. Registering and Aligning Bands

From Preprocessing:
- Use "Register Bands"
- Select reference band
- Use ROI selector
- Apply alignment methods (correlation, SuperGlue, ECC)

Outputs:
- Aligned cube
- Transformation matrices stored in JSON

---

## 4. Spectral Analysis Tools

In the Spectral Analysis tab:
- Select a pixel or region
- Plot reflectance curves
- Apply smoothing (Savitzky–Golay)
- Export spectral profiles

---

## 5. RGB Exploration

Tools include:
- Rectangle-based sampling
- Visual color mapping
- Mean RGB extraction
- Quick comparisons between regions

---

## 6. GeoImport Workflow

The GeoImport module allows:
- Importing drone image folders
- Running OpenDroneMap reconstruction (requires Docker)
- Auto-generating orthomosaics
- Aligning orthomosaics with panels
- Exporting calibrated geospatial cubes

Outputs are saved in:
<folder>/__odm_outputs/

---

## 7. Saving Results

You can export:
- NPY cubes
- Aligned cubes
- Spectral plots (PNG)
- JSON metadata

---

## 8. Troubleshooting

If images fail to load:
- Confirm file format is supported
- Verify wavelengths JSON is present

If alignment fails:
- Try a different reference band
- Reduce ROI size
- Use SuperGlue alignment

If GUI freezes:
- Avoid extremely large cubes (>2 GB)
- Increase system RAM

---

For installation help, read:
docs/INSTALL_en.md
