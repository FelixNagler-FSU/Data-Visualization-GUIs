import os
import time
import shutil
from datetime import datetime, timedelta
from pywinauto.application import Application
from pywinauto.keyboard import send_keys
import pyautogui


# --- Benutzerdefinierte Variablen ---
EC_LAB_PATH = r"C:\Program Files (x86)\EC-Lab\11.62\EClab.exe"
DATA_DIRECTORY = r"C:\Users\ro45vij\Desktop\AA_Data-Processing\ZZ_Biologic-Export"
PLUS_BUTTON_IMAGE = r"C:\Users\ro45vij\Pictures\Plus-in-EC-Lab.PNG"
OPEN_BUTTON_IMAGE = r"C:\Users\ro45vij\Pictures\Open-Button-in-EC-Lab.PNG"

def get_recent_mpr_files(directory, minutes=60):
    """
    Liefert alle .mpr-Dateien im angegebenen Verzeichnis zurück,
    die innerhalb der letzten `minutes` Minuten geändert wurden.
    """
    cutoff = datetime.now() - timedelta(minutes=minutes)
    recent_files = []

    for fname in os.listdir(directory):
        if fname.lower().endswith(".mpr"):
            fpath = os.path.join(directory, fname)
            mtime = datetime.fromtimestamp(os.path.getmtime(fpath))
            if mtime >= cutoff:
                recent_files.append(fname)

    return recent_files

def find_and_copy_modified_files(src_dir, dest_dir, ext):
    """
    Diese Funktion durchsucht das angegebene Quellverzeichnis (src_dir) rekursiv
    nach Dateien mit der definierten Endung (ext). Sie kopiert die Dateien nur dann
    in den Zielordner, wenn sie im Quellordner neuer sind als eine bestehende .mpr-Datei
    mit gleichem Namen im Zielordner. Wenn keine .mpr-Datei existiert, wird die .mpr-Datei
    kopiert.
    """
    copied_files_count = 0
    copied_files_list = []

    if not os.path.exists(dest_dir):
        print(f"Zielverzeichnis '{dest_dir}' existiert nicht. Es wird erstellt.")
        os.makedirs(dest_dir)

    print(f"Suche nach '{ext}' Dateien im Verzeichnis: {src_dir}")
    print("--------------------------------------------------")

    for root, dirs, files in os.walk(src_dir):
        for filename in files:
            if filename.lower().endswith(ext.lower()):
                source_filepath = os.path.join(root, filename)
                base_filename = os.path.splitext(filename)[0]
                destination_mpr_filepath = os.path.join(dest_dir, f"{base_filename}.mpr")
                source_mod_time = datetime.fromtimestamp(os.path.getmtime(source_filepath))
                should_copy = False

                if os.path.exists(destination_mpr_filepath):
                    destination_mod_time = datetime.fromtimestamp(os.path.getmtime(destination_mpr_filepath))
                    if source_mod_time > destination_mod_time:
                        should_copy = True
                        print(f"Gefunden: {filename}")
                        print(f"--> Quelldatei ist neuer ({source_mod_time} > {destination_mod_time}).")
                    else:
                        print(f"Überspringe: {filename} (Quelldatei ist nicht neuer als die bestehende Datei)")
                else:
                    should_copy = True
                    print(f"Gefunden: {filename} (keine bestehende Datei, wird kopiert)")

                if should_copy:
                    try:
                        shutil.copy2(source_filepath, dest_dir)
                        print(f"--> Kopiert nach: {dest_dir}")
                        copied_files_list.append(filename)
                        copied_files_count += 1
                    except Exception as e:
                        print(f"--> Fehler beim Kopieren von {filename}: {e}")

    print("--------------------------------------------------")
    print(copied_files_list)
    print(f"Vorgang abgeschlossen. {copied_files_count} Datei(en) wurden kopiert.")
    return copied_files_count




if __name__ == "__main__":
    # Beispiel: erst neue Dateien kopieren, dann EC-Lab starten
    SRC_DIR = r"S:\Group Members\Felix_Nagler\Raw-Data_Biologic"
    DEST_DIR = DATA_DIRECTORY

    copied_count = find_and_copy_modified_files(SRC_DIR, DEST_DIR, ".mpr")
    print(f"\n{copied_count} neue Dateien wurden vorbereitet.\n")

   
