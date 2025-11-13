import os
import pickle
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

# Delay importing heavy automation modules (pywinauto/pyautogui) by lazily importing
# DatenExport only when the export is triggered. This avoids potential interference
# with the native file dialogs causing hangs when clicking Browse.
_export_func = None

def _get_export_func():
    global _export_func
    if _export_func is not None:
        return _export_func
    try:
        from DatenExport import find_and_copy_modified_files as f
        _export_func = f
        return _export_func
    except Exception as e:
        raise RuntimeError(f"Failed to import find_and_copy_modified_files from DatenExport.py: {e}")


SETTINGS_FILE = Path(__file__).with_name("last_settings_data_export.pkl")


class DataExportGUI(tk.Tk):
    """Simple GUI to copy measurement files from a Source directory to a Destination directory.

    Behavior:
    - Asks for Source and Destination directories.
    - Optional file extension filter (default: .mpr).
    - Runs the copy routine from DatenExport.find_and_copy_modified_files.
    - Persists last used paths across sessions.
    """

    def __init__(self):
        super().__init__()
        self.title("Data Export")
        self.geometry("700x260")
        self.resizable(False, False)

        self._build_ui()
        self._load_settings()

    def _build_ui(self):
        pad_x = 10
        pad_y = 8

        # Source directory
        tk.Label(self, text="Source Directory:").grid(row=0, column=0, sticky="e", padx=pad_x, pady=pad_y)
        self.src_entry = tk.Entry(self, width=70)
        self.src_entry.grid(row=0, column=1, sticky="w", padx=(0, pad_x), pady=pad_y)
        tk.Button(self, text="Browse...", command=self._browse_src).grid(row=0, column=2, padx=(0, pad_x), pady=pad_y)

        # Destination directory
        tk.Label(self, text="Destination Directory:").grid(row=1, column=0, sticky="e", padx=pad_x, pady=pad_y)
        self.dest_entry = tk.Entry(self, width=70)
        self.dest_entry.grid(row=1, column=1, sticky="w", padx=(0, pad_x), pady=pad_y)
        tk.Button(self, text="Browse...", command=self._browse_dest).grid(row=1, column=2, padx=(0, pad_x), pady=pad_y)

        # Extension filter
        tk.Label(self, text="File Extension:").grid(row=2, column=0, sticky="e", padx=pad_x, pady=pad_y)
        self.ext_entry = tk.Entry(self, width=10)
        self.ext_entry.insert(0, ".mpr")
        self.ext_entry.grid(row=2, column=1, sticky="w", padx=(0, pad_x), pady=pad_y)

        # Action buttons
        tk.Button(self, text="Run Export", command=self._run_export, width=18).grid(row=3, column=1, sticky="w", padx=(0, pad_x), pady=(pad_y, 4))
        tk.Button(self, text="Open Destination", command=self._open_dest, width=18).grid(row=3, column=1, sticky="w", padx=(140, pad_x), pady=(pad_y, 4))

        # Status label
        self.status_var = tk.StringVar(value="Ready.")
        self.status_label = tk.Label(self, textvariable=self.status_var, anchor="w")
        self.status_label.grid(row=4, column=0, columnspan=3, sticky="we", padx=pad_x, pady=(pad_y, 0))

    def _browse_src(self):
        directory = filedialog.askdirectory(title="Select Source Directory")
        if directory:
            self.src_entry.delete(0, tk.END)
            self.src_entry.insert(0, directory)

    def _browse_dest(self):
        directory = filedialog.askdirectory(title="Select Destination Directory")
        if directory:
            self.dest_entry.delete(0, tk.END)
            self.dest_entry.insert(0, directory)

    def _validate(self):
        src = self.src_entry.get().strip()
        dest = self.dest_entry.get().strip()
        ext = self.ext_entry.get().strip() or ".mpr"

        if not src:
            messagebox.showerror("Validation Error", "Please select a Source directory.")
            return None
        if not os.path.isdir(src):
            messagebox.showerror("Validation Error", "Source directory does not exist.")
            return None
        if not dest:
            messagebox.showerror("Validation Error", "Please select a Destination directory.")
            return None
        # dest can be created by the export function; we don't enforce existence here
        if not ext.startswith('.'):
            ext = f".{ext}"
        return src, dest, ext

    def _run_export(self):
        validated = self._validate()
        if not validated:
            return
        src, dest, ext = validated

        # Persist settings early
        self._save_settings()

        # Disable buttons during run
        self._set_buttons_state(tk.DISABLED)
        self.status_var.set("Export running...")
        self.update_idletasks()

        def worker():
            try:
                export_func = _get_export_func()
                count = export_func(src, dest, ext)
                def on_success():
                    self.status_var.set(f"Done. {count} file(s) copied.")
                    messagebox.showinfo("Export Complete", f"{count} file(s) copied to:\n{dest}")
                    self._set_buttons_state(tk.NORMAL)
                self.after(0, on_success)
            except Exception as e:
                def on_fail():
                    self.status_var.set("Error during export.")
                    messagebox.showerror("Export Failed", str(e))
                    self._set_buttons_state(tk.NORMAL)
                self.after(0, on_fail)

        threading.Thread(target=worker, daemon=True).start()

    def _set_buttons_state(self, state):
        # Iterate children to adjust buttons
        for child in self.children.values():
            if isinstance(child, tk.Button):
                child.configure(state=state)
        # Also grid_slaves if buttons nested
        for widget in self.grid_slaves():
            if isinstance(widget, tk.Button):
                widget.configure(state=state)

    def _open_dest(self):
        dest = self.dest_entry.get().strip()
        if dest and os.path.isdir(dest):
            try:
                os.startfile(dest)  # Windows only
            except Exception as e:
                messagebox.showerror("Open Destination Failed", str(e))
        else:
            messagebox.showwarning("Open Destination", "Destination directory is not set or does not exist.")

    def _save_settings(self):
        settings = {
            "src": self.src_entry.get().strip(),
            "dest": self.dest_entry.get().strip(),
            "ext": self.ext_entry.get().strip() or ".mpr",
        }
        try:
            with open(SETTINGS_FILE, "wb") as f:
                pickle.dump(settings, f)
        except Exception:
            # Non-fatal if settings can't be saved
            pass

    def _load_settings(self):
        if not SETTINGS_FILE.exists():
            return
        try:
            with open(SETTINGS_FILE, "rb") as f:
                settings = pickle.load(f)
            if isinstance(settings, dict):
                self.src_entry.delete(0, tk.END)
                self.src_entry.insert(0, settings.get("src", ""))
                self.dest_entry.delete(0, tk.END)
                self.dest_entry.insert(0, settings.get("dest", ""))
                self.ext_entry.delete(0, tk.END)
                self.ext_entry.insert(0, settings.get("ext", ".mpr"))
        except Exception:
            # Ignore load errors and start fresh
            pass


if __name__ == "__main__":
    app = DataExportGUI()
    app.mainloop()
