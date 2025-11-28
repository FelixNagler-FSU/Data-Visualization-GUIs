# Data-Visualization-GUIs

From Biologic (.mpr) battery test data to reproducible Matplotlib visualizations and structured exports (TXT, NetCDF, ZIP) via multiple Tkinter GUIs.

---

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

## Quick Setup Guide

Follow these steps in order to get started from scratch:

### 1. Download and Install Python 3.12

**Download:**
- Go to https://www.python.org/downloads/
- Download **Python 3.12.x** (latest stable version)

**Installation Location:**
- The installer will suggest: `C:\Users\<YourUsername>\AppData\Local\Programs\Python\Python312`
- This is the **recommended location** (default)

**Installation Steps:**
1. Run the downloaded installer
2. ✅ **CRITICAL:** Check **"Add Python to PATH"** at the bottom of the first screen
3. Click **"Install Now"**
4. Wait for installation to complete
5. Click **"Close"**

**Verify Installation:**
Open PowerShell (Windows key + type "PowerShell") and run:
```powershell
python --version
```
Should show: `Python 3.12.x`

If it says "command not found", Python was not added to PATH. Reinstall and check the PATH option.

---

### 2. Download and Install Visual Studio Code

**Download:**
- Go to https://code.visualstudio.com/
- Download the **Windows 64-bit** version

**Installation Steps:**
1. Run the installer
2. ✅ **Recommended options to select:**
   - ✓ Add "Open with Code" action to Windows Explorer context menu
   - ✓ Add "Open with Code" action to directory context menu
   - ✓ Add to PATH (enables `code .` command in terminal)
   - ✓ Register Code as an editor for supported file types
3. Click **"Next"** → **"Install"**
4. Launch VS Code when installation completes

**Install Python Extension:**
1. In VS Code, click the **Extensions** icon in the left sidebar (or press `Ctrl+Shift+X`)
2. Search for **"Python"**
3. Install **"Python"** extension by Microsoft
4. The **"Pylance"** extension will install automatically (language server for Python)

---

### 3. Download the Git Repository

**Recommended Save Location:**
```
C:\Dev\Data-Visualization-GUIs
```

**Why this specific location?**
- ✅ Short path (avoids Windows 260 character path limit)
- ✅ No spaces (prevents issues with some tools)
- ✅ Not in OneDrive/Dropbox (prevents file locking during data export)
- ✅ Not in Documents (avoids backup/sync conflicts)
- ✅ Easy to type and remember

**Method A: Using Git (Recommended)**

If you don't have Git installed:
1. Download Git from: https://git-scm.com/download/win
2. Run installer with default options
3. Restart PowerShell after installation

Then download the repository:
```powershell
# Create the Dev folder
mkdir C:\Dev

# Navigate into it
cd C:\Dev

# Clone the repository
git clone https://github.com/FelixNagler-FSU/Data-Visualization-GUIs.git

# Enter the project folder
cd Data-Visualization-GUIs
```

**Method B: Download ZIP (Alternative)**

1. Go to: https://github.com/FelixNagler-FSU/Data-Visualization-GUIs
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Extract the ZIP file to `C:\Dev\Data-Visualization-GUIs`
   - Right-click the ZIP → "Extract All..."
   - Choose `C:\Dev` as destination
   - Rename folder to `Data-Visualization-GUIs` if needed

---

### 4. Open Project in Visual Studio Code

**Option A: From PowerShell**
```powershell
cd C:\Dev\Data-Visualization-GUIs
code .
```

**Option B: From VS Code**
1. Open VS Code
2. Click **File** → **Open Folder...**
3. Navigate to `C:\Dev\Data-Visualization-GUIs`
4. Click **"Select Folder"**

---

### 5. Select Python Interpreter in VS Code

1. Press `Ctrl+Shift+P` to open the **Command Palette**
2. Type: **"Python: Select Interpreter"**
3. Select your Python 3.12 installation
   - Should show path like: `C:\Users\...\AppData\Local\Programs\Python\Python312\python.exe`
   - If multiple versions appear, choose **3.12.x**

---

### 6. Create Virtual Environment and Install Dependencies

A virtual environment isolates this project's packages from your system Python.

**In VS Code:**
1. Open a new terminal: Press `Ctrl+Shift+ö` (or **Terminal** → **New Terminal**)
2. Make sure you're in the project folder (should show `C:\Dev\Data-Visualization-GUIs`)

**Run these commands one by one:**

```powershell
# Create virtual environment in .venv folder
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install all required packages from requirements.txt
pip install -r requirements.txt
```

**If you get an execution policy error** when activating, run this first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then try activating again.

**Verify packages were installed:**
```powershell
pip list
```
Should show packages like `matplotlib`, `pandas`, `numpy`, `yadg`, etc.

**After installation, re-select the interpreter:**
1. Press `Ctrl+Shift+P`
2. Type: **"Python: Select Interpreter"**
3. Choose the one with `.venv` in the path:
   - `.\\.venv\\Scripts\\python.exe` (Recommended)

---

### 7. Run the GUIs

**Make sure your virtual environment is activated!**
- Your terminal prompt should show `(.venv)` at the beginning
- If not, run: `.\.venv\Scripts\Activate.ps1`

**Run any GUI:**
```powershell
python GUI_capa-cycle.py
python GUI_voltage-capa.py
python GUI_dQ-dE.py
python GUI_data_export.py
```

The GUI window will open. Close it when done - the terminal will be ready for the next command.

---

## Daily Workflow (After Initial Setup)

Every time you work on this project:

1. **Open VS Code**
   - Double-click the folder `C:\Dev\Data-Visualization-GUIs` or
   - In VS Code: File → Open Recent → Data-Visualization-GUIs

2. **Open Terminal** (if not already open)
   - Press `Ctrl+Shift+ö` or Terminal → New Terminal

3. **Activate Virtual Environment** (if `(.venv)` is not shown in terminal prompt)
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

4. **Run your GUI**
   ```powershell
   python GUI_capa-cycle.py
   ```

5. **When done** (optional - closes virtual environment)
   ```powershell
   deactivate
   ```

---

## Requirements Files Explained

Two requirements files are provided:

| File | Purpose |
|------|---------|
| `requirements.txt` | **Pinned versions** (`package==1.2.3`). Use this for reproducible results. |
| `requirements-no-version.txt` | **Unpinned versions** (`package`). Gets latest compatible versions. |

**Why pinned versions?**
- ✅ Identical behavior across different computers
- ✅ Results stay reproducible over time
- ✅ Prevents breaking changes from package updates

**When to update requirements.txt:**
After adding a new package:
```powershell
pip install new-package-name
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Add new-package-name"
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `python: command not found` | Python not added to PATH during installation. Reinstall Python and check "Add to PATH". |
| Cannot activate `.venv` | Run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` |
| VS Code uses wrong Python | Press `Ctrl+Shift+P` → "Python: Select Interpreter" → Choose `.venv\Scripts\python.exe` |
| `ModuleNotFoundError: No module named 'X'` | Virtual environment not activated or packages not installed. Run: `.\.venv\Scripts\Activate.ps1` then `pip install -r requirements.txt` |
| GUI window doesn't appear | Check that matplotlib backend is TkAgg (already set in code). Try running: `python -c "import tkinter; tkinter.Tk()"` |
| Long path errors when cloning | Move repository to `C:\Dev` (shorter path). Or enable long paths: `git config --system core.longpaths true` |
| "Cannot find .venv" | You're in wrong directory. Run: `cd C:\Dev\Data-Visualization-GUIs` |
| GUI freezes during export | Make sure you're using updated `GUI_data_export.py` with threading support |

---

## Git Basics (Optional - for Contributing)

If you want to save your changes or contribute back to the repository:

```powershell
# See what files you changed
git status

# Stage all changes
git add .

# Commit with a message
git commit -m "Describe what you changed"

# Push to GitHub (requires write access)
git push origin 251022_code-development
```

**For first-time Git users:**
Configure your identity (one time):
```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

---

## Advanced: Using Conda/Miniforge (Optional)

If you need better management of compiled libraries (like `netCDF4`, `h5py`) or want multi-language support, consider Miniforge:

**Download:** https://github.com/conda-forge/miniforge#download

**Create environment:**
```powershell
conda create -n dv-guis -c conda-forge python=3.12
conda activate dv-guis
pip install -r requirements.txt
```

**Note:** For most users, the standard Python + venv setup (steps 1-7 above) is sufficient and simpler.

---

## License

See `LICENSE` for details.

---

*Last updated: 2025-11-19 (streamlined for first-time users)*
