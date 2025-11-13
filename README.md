# Data-Visualization-GUIs

From Biologic (.mpr) battery test data to reproducible Matplotlib visualizations and structured exports (TXT, NetCDF, ZIP) via multiple Tkinter GUIs.

## Overview

This repository contains several Python GUI tools to visualize and export Biologic cycling data:

| GUI | Purpose |
| --- | ------- |
| `GUI_capa-cycle.py` | Specific capacity / coulombic efficiency vs cycle, NetCDF + figures |
| `GUI_voltage-capa.py` | Voltage vs specific capacity per selected cycles |
| `GUI_dQ-dE.py` | dQ/dE vs voltage with optional Savitzky–Golay smoothing |
| `open_pickle.py` | Re-open previously pickled Matplotlib figure objects |
| `GUI_data_export.py` | Copy newly modified `.mpr` files from a source to destination folder |

Each visualization GUI supports selectable save options (PNG, PDF, SVG, TXT, NetCDF, ZIP) and persists your last settings between runs.

---
## Python Installation Options

You only need one working Python distribution. Choose ONE of:

1. Official Python installer (https://www.python.org/downloads/) – simple, fine for venv usage.
2. Miniforge (recommended for data/science) – lightweight Conda distribution with clean dependency management.

If you already have Python installed (via VS Code prompt or official installer) you can skip Miniforge. If you want easier binary package management (e.g. `netCDF4`, `h5py`), use Miniforge.

### Miniforge Steps (Optional but Recommended)
Download: https://github.com/conda-forge/miniforge#download (get `Miniforge3-Windows-x86_64.exe`)

Installer recommendations:
* Install for "Just Me"
* Path: `C:\Miniforge3`
* Add to PATH (optional) – otherwise use the Miniforge Prompt

Verify:
```powershell
conda --version
python --version
```

Optional speed‑up:
```powershell
conda install -n base -c conda-forge mamba
```

---
## Best Practice: Global vs Project Environments

| Scope | Use Case | Example Name |
|-------|----------|--------------|
| Global Conda env | Shared interactive data analysis tools (Jupyter, linters) | `analysis` |
| Project-specific env (conda or venv) | Reproducible development & deployment | `.venv` or `dv-guis` |

Guidelines:
1. Keep the `base`/global env lean; avoid installing project-specific libs there.
2. Use a dedicated environment per repository for exact dependency versions.
3. Pin versions (as in `requirements.txt`) for reproducibility; avoid mixing conda and pip in the same env unless necessary.
4. Prefer a virtual environment (`venv`) when you only need pure-Python + pip packages; prefer Conda when you rely on compiled libraries, alternative BLAS, or system tools.

---
## Create an Environment (Choose One Method)

### Conda (via Miniforge)
```powershell
conda create -n dv-guis -c conda-forge python=3.12
conda activate dv-guis
pip install -r requirements.txt
```
Unpinned first then freeze:
```powershell
pip install -r requirements-no-version.txt
pip freeze > requirements.txt
```
Pros: Good for complex native deps. Cons: Avoid mixing many conda + pip upgrades to prevent conflicts.

---
### venv (Official Python Installer)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Unpinned then freeze:
```powershell
pip install -r requirements-no-version.txt
pip freeze > requirements.txt
```
Deactivate: `deactivate` | Re-activate: `.\.venv\Scripts\Activate.ps1`

---
## Running the GUIs

Activate your environment first (conda or venv). Then run any script:
```powershell
python GUI_capa-cycle.py
python GUI_voltage-capa.py
python GUI_dQ-dE.py
python GUI_data_export.py
```

Plots use the TkAgg backend (requires a working Tk installation—bundled with standard Python on Windows). Close the Matplotlib window to return focus to the GUI without terminating the app.

---
## Requirements Files Explained

Two files are provided:

1. `requirements.txt` – Pinned versions (exact `==`) for fully reproducible installs. Use this for production, sharing with collaborators, or archiving analysis states.
2. `requirements-no-version.txt` – Unpinned, minimal spec. Convenient for quick experimentation or getting the latest compatible releases, but not reproducible.

Why use pinned versions?
* Guarantee identical behavior across machines and time.
* Prevent silent API changes from breaking scientific results.
* Enable reliable bug reports and regression tests.

When to regenerate `requirements.txt`:
* After intentional upgrades: Install new versions, run your test workflows, then `pip freeze > requirements.txt`.
* After removing packages: Clean with `pip uninstall ...`, validate, then freeze.

Suggested workflow:
1. Start from `requirements-no-version.txt` in a fresh env.
2. Freeze to `requirements.txt` once stable.
3. Commit both files; treat `requirements.txt` as authoritative for reproducibility.

---
## Updating/Adding Dependencies

To add a new library (example: `netCDF4`):
```powershell
pip install netCDF4
pip freeze > requirements.txt
```

For Conda (if binary reliability matters):
```powershell
conda activate dv-guis
conda install -c conda-forge netcdf4
pip freeze > requirements.txt  # still captures pip-installed packages; conda packages appear as versions resolved.
```

---
## Troubleshooting

| Issue | Suggestion |
|-------|------------|
| GUI window unresponsive during export | Confirm you are using the updated `GUI_data_export.py` with threading. |
| Backend errors (TkAgg) | Ensure you are using CPython from Miniforge (not a stripped-down embedded Python). |
| Version conflict after mixing conda/pip | Consider recreating env and using only pip with pinned `requirements.txt`. |
| Different results vs colleague | Verify both used the same `requirements.txt` commit hash. |

---
## Reproducibility Checklist

Before sharing results or archiving:
1. Commit the current `requirements.txt`.
2. Record Python version: `python --version`.
3. Note OS & architecture (Windows x86_64).
4. Optionally export full env details:
	```powershell
	pip freeze > requirements_frozen.txt
	```
## Python & Environment (Condensed)

Assumption: You install Python 3.12 when VS Code first prompts you (or from https://www.python.org). No Conda needed.
---
Create a virtual environment once per clone:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Re-activate later:
```powershell
.\.venv\Scripts\Activate.ps1
```
Deactivate:
```powershell
deactivate
```
## License

See `LICENSE` for details.

Conda path:
```powershell
conda create -n dv-guis -c conda-forge python=3.11
conda activate dv-guis
pip install -r requirements.txt
python GUI_capa-cycle.py
```

venv path:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python GUI_voltage-capa.py
```

---
## Getting the Repository

### Option 1: Download ZIP
On the GitHub page click "Code" > "Download ZIP", extract to a short path (e.g. `C:\Dev\Data-Visualization-GUIs`).
Activate the venv first, then run any script:

### Option 2: Git Clone (Recommended)

Install Git if you don't have it: https://git-scm.com/download/win

Recommended base directory (avoid OneDrive sync issues & long paths):
```
C:\Dev\    (create if it does not exist)
```

Then clone:
```powershell
cd C:\Dev
git clone https://github.com/FelixNagler-FSU/Data-Visualization-GUIs.git
cd Data-Visualization-GUIs
```

If you plan to contribute, optionally create a fork first and clone your fork URL.


### Avoid Storing Here
* Deep nested paths with spaces (e.g. `C:\Users\<name>\Documents\Some Folder\Another Folder\...`) can cause path length issues.
* Cloud sync folders (OneDrive, Dropbox) may lock files while Python writes figures or NetCDF files.

---
## Visual Studio Code Setup

### 1. Install VS Code
Download: https://code.visualstudio.com/
Run installer and select:
* Add "Open with Code" to context menu (optional convenience)
* Add VS Code to PATH (so `code .` works in PowerShell)

### 2. Open the Project
```powershell
cd C:\Dev\Data-Visualization-GUIs
code .
```

### 3. Install Extensions
In VS Code (left sidebar Extensions):
* Python (Microsoft)
* Pylance
* GitLens (optional for history)
* (Optional) Jupyter if you plan notebooks

### 4. Select Interpreter
Press `Ctrl+Shift+P` → "Python: Select Interpreter" → choose your conda env (`dv-guis`) or `.venv` path.

### 5. Create venv (If Not Done)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Then re-select the interpreter pointing to `.venv`.
```

### 7. Git Basics
git clone https://github.com/FelixNagler-FSU/Data-Visualization-GUIs.git
cd Data-Visualization-GUIs
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python GUI_capa-cycle.py
```
```powershell
git status
git add README.md
git commit -m "Update README with setup details"
git push origin 251022_code-development
```

### 8. Common First-Time Issues
| Symptom | Fix |
|---------|-----|
| VS Code uses wrong Python | Re-select interpreter; ensure environment activated in terminal. |
| Cannot import packages | Verify `pip list` inside the selected interpreter shows dependencies. |
| Long path errors | Move repo to `C:\Dev` or enable long paths in Git (`git config --system core.longpaths true`). |

---
*Last updated: 2025-11-13 (deduplicated setup instructions)*
