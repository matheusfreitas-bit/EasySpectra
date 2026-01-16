# Installing EasySpectra

This document describes how to install and run EasySpectra on a clean system.

---

## 🖥️ System Requirements

- Python **>= 3.10** (recommended: Python 3.11)
- Operating system:
  - macOS
  - Linux
  - Windows

- **Docker** (required for orthomosaic generation):
  - https://www.docker.com/get-started

---

## Installation

🐍 Step 1 — Install Python

Download and install Python from:

```
https://www.python.org/downloads/
```

Make sure Python is available in the terminal:

```Copy code
python --version
```
or
```Copy code
python3 --version
```
📥 Step 2 — Clone the EasySpectra repository

```Copy code
git clone https://github.com/matheusfreitas-bit/EasySpectra.git
cd EasySpectra
```

🧪 Step 3 — Create and activate a virtual environment
macOS / Linux

```Copy code
python3 -m venv .venv
source .venv/bin/activate
Windows (PowerShell)
```

```Copy code
python -m venv .venv
.venv\Scripts\activate
```

⬆️ Step 4 — Upgrade pip

```Copy code
pip install --upgrade pip
📦 Step 5 — Install Python dependencies
```

```Copy code
pip install -r requirements.txt
```

>⚠️ Important note about PyTorch:
>This project uses torch. If you want GPU support or encounter installation issues, please follow the official instructions at:
>https://pytorch.org/get-started/locally/

🐳 Step 6 — Install and configure Docker (for orthomosaic)
EasySpectra uses OpenDroneMap (ODM) via Docker to generate orthomosaics.

Install Docker:
```
https://www.docker.com/get-started
```
Check Docker installation:

```Copy code
docker --version
```

Download the ODM image:

```Copy code
docker pull opendronemap/odm
```

>⚠️ Without Docker and ODM, the orthomosaic (GeoImport) step will not run.

▶️ Step 7 — Run EasySpectra
From the project root directory:

```Copy code
python launcher.py
```
On macOS, you may also use:

```Copy code
./EasySpectra.command
```

🧪 Troubleshooting

Problem: ModuleNotFoundError

>Make sure:
>The virtual environment is activated

>You installed dependencies with:

```Copy code
pip install -r requirements.txt
```

Problem: Docker not found

>Check:
```Copy code
docker --version
```

>If not found, install Docker from:

```
https://www.docker.com/get-started
```

Problem: Permission denied on macOS for Docker
>Make sure Docker Desktop is running before launching EasySpectra.

✅ Installation Complete
You are now ready to use EasySpectra.

