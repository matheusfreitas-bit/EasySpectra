# EasySpectra

**EasySpectra** is an integrated Python platform for multispectral and hyperspectral image processing and analysis, designed to provide a complete and user-friendly workflow from image preprocessing to spectral analysis and data export. The software is especially oriented to applications in weed science, precision agriculture, and plant phenotyping.

---

## 🚀 Main Features

- 📥 Import multispectral and hyperspectral images (e.g., TIFF and spectral cubes)
- 🔄 Band alignment and geometric correction (including advanced methods such as SuperGlue)
- 🎯 Radiometric calibration and reflectance normalization
- 🖼️ Graphical user interface (GUI) based on Qt (PySide6)
- ✏️ Region of interest (ROI) selection (rectangular, circular, and freehand)
- 📊 Spectral signature extraction and visualization
- 🌱 Vegetation index computation (e.g., NDVI, GNDVI)
- 📁 Export of results to CSV for statistical analysis and modeling
- 🧩 Modular architecture for future extensions (e.g., machine learning)

---

## 🖥️ System Requirements

- Python **>= 3.10** (recommended: Python 3.11)
- Operating systems:
  - macOS
  - Linux
  - Windows

---

## 📦 Installation

It is **strongly recommended** to use a Python virtual environment.

### 1️⃣ Clone the repository

```bash
git clone https://github.com/matheusfreitas-bit/EasySpectra.git
cd EasySpectra
```

2️⃣ Create and activate a virtual environment
macOS / Linux:

```Copy code
python3 -m venv .venv
source .venv/bin/activate
```
Windows (PowerShell):
powershell
```Copy code
python -m venv .venv
.venv\Scripts\activate
```

3️⃣ Upgrade pip
```Copy code
pip install --upgrade pip
```

4️⃣ Install dependencies
```bash
Copy code
pip install -r requirements.txt
```

⚠️ Important note about PyTorch:
This project uses torch. If you have issues installing it or want GPU support, please follow the official instructions at:
https://pytorch.org/get-started/locally/

▶️ Running EasySpectra
To launch the graphical interface, run:

```bash
Copy code
python launcher.py
```

Alternatively, you can run the main module directly (advanced use):

```bash
Copy code
python -m easyspectra
```

On macOS, you may also use:

```bash
Copy code
./EasySpectra.command
```

🧭 Typical Workflow
A typical workflow in EasySpectra consists of:

Import multispectral or hyperspectral image data

Align spectral bands and apply geometric correction

Perform radiometric calibration using reference panels

Define regions of interest (ROI) interactively

Extract spectral signatures

Compute vegetation indices (e.g., NDVI, GNDVI)

Export results to CSV for further statistical analysis

📁 Project Structure
```text
Copy code
EasySpectra/
 ├── src/easyspectra/      # Main source code
 ├── deep_models/         # Deep learning models (e.g., SuperGlue)
 ├── icons/               # GUI icons
 ├── docs/                # Documentation
 ├── launcher.py          # Main launcher script
 ├── requirements.txt     # Python dependencies
 ├── pyproject.toml       # Project configuration
 └── README.md
```

📊 Dataset Example
Some examples and figures were generated using the WeedCube dataset:

Ram, B. G., Mettler, J., Howatt, K., Ostlie, M., & Sun, X. (2024).
WeedCube: Proximal hyperspectral image dataset of crops and weeds for machine learning applications.
Data in Brief, 56, 110837.
https://doi.org/10.1016/j.dib.2024.110837

🧠 Scientific Motivation
EasySpectra was developed to reduce technical barriers in spectral image analysis by integrating the entire workflow into a single, accessible platform, allowing researchers to focus on experimental design, interpretation, and decision-making rather than on complex data preprocessing steps.

🔮 Future Developments
Planned features include:

Integration of machine learning classifiers

Batch processing pipelines

Tools for large-scale field experiments

Additional vegetation indices and spectral tools

📜 License
This project is currently distributed under proprietary terms.
A specific open-source license may be added in the future.

🤝 Contributing
Contributions, suggestions, and bug reports are welcome.
Please use the GitHub Issues page to report problems or request features.

📬 Contact
If you use EasySpectra in your research, please cite the software and feel free to contact the author.
