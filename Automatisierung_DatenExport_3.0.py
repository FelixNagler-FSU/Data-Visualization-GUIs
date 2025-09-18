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


def automate_ec_lab():
    try:
        print(f"Starte EC-Lab von: {EC_LAB_PATH}")
        app = Application(backend="win32").start(EC_LAB_PATH)
        time.sleep(8)

        main_window = app.top_window()
        main_window.wait('ready', timeout=20)
        main_window.wait_for_idle()
        print("Verbunden mit dem Hauptfenster von EC-Lab.")

        print("Sende Tastenkombination 'Strg+T'...")
        main_window.set_focus()
        send_keys('^t')
        time.sleep(2)

        print("Warte auf Fenster 'Text File Export'...")
        export_window = app.window(title_re=".*Text File Export.*")
        export_window.wait('ready', timeout=30)
        print("Verbunden mit dem Fenster 'Text File Export'.")

        print(f"Suche nach der Plus-Schaltfläche ({PLUS_BUTTON_IMAGE})...")
        plus_button_location = pyautogui.locateOnScreen(PLUS_BUTTON_IMAGE, confidence=0.9)

        if plus_button_location:
            print("Plus-Schaltfläche gefunden. Klicke darauf...")
            pyautogui.click(plus_button_location)
            time.sleep(2)
        else:
            print("Fehler: Plus-Schaltfläche nicht gefunden.")
            return

        print("Warte auf das 'Öffnen'-Dialogfenster...")
        # Flexiblere Erkennung für "Open" oder "Öffnen"
        open_dialog = app.window(title_re="\\*?(Open|Öffnen)")
        open_dialog.wait('ready', timeout=10)
        print("Verbunden mit dem 'Öffnen'-Dialog.")

        file_path_edit = open_dialog.child_window(class_name="Edit")
        file_path_edit.wait('ready', timeout=5)
        print(f"Gebe den Dateipfad ein: {DATA_DIRECTORY}")
        file_path_edit.set_text(DATA_DIRECTORY)
        send_keys('{ENTER}')
        time.sleep(2)

        # --- NEU: nur Dateien der letzten Stunde auswählen ---
        recent_files = get_recent_mpr_files(DATA_DIRECTORY, minutes=60)

        if not recent_files:
            print("Keine neuen .mpr-Dateien in der letzten Stunde gefunden. Abbruch.")
            return

        # Format: "file1.mpr" "file2.mpr" "file3.mpr"
        files_to_open = " ".join([f'"{f}"' for f in recent_files])
        print(f"Öffne Dateien: {files_to_open}")

        file_path_edit = open_dialog.child_window(class_name="Edit")
        file_path_edit.wait('ready', timeout=5)
        file_path_edit.set_text(files_to_open)
        send_keys('{ENTER}')
        time.sleep(2)


        #print("Wähle alle Dateien mit 'Strg+A' aus...")
        #file_list_view = open_dialog.child_window(class_name="DUIViewWndClassName")
        #file_list_view.wait('ready', timeout=5)
        #rect = file_list_view.rectangle()
        #pyautogui.click((rect.left + rect.right) / 2, (rect.top + rect.bottom) / 2)
        #time.sleep(0.5)
        #send_keys('^a')
        #time.sleep(1)

        # Open/Öffnen-Button klicken
        try:
            open_button = None
            for title in ["Open", "&Open", "Öffnen", "&Öffnen"]:
                try:
                    open_button = open_dialog.child_window(title=title, class_name="Button")
                    open_button.wait('visible enabled', timeout=5)
                    open_button.click_input()
                    print(f"'{title}'-Button erfolgreich geklickt.")
                    break
                except Exception:
                    button_location = pyautogui.locateOnScreen(OPEN_BUTTON_IMAGE, confidence=0.9)
                    if button_location:
                        pyautogui.click(button_location)
                        print("Fallback: 'Open'-Button per Bildschirmerkennung geklickt.")
                        break
            else:
                print("Fehler: Kein 'Open/Öffnen'-Button gefunden.")
                return
        except Exception as e:
            print(f"Fehler beim Klicken auf 'Open': {e}")
            return

        print("Dateien erfolgreich in die Liste zur Konvertierung hinzugefügt.")
    except Exception as e:
        print(f"Es ist ein Fehler aufgetreten: {e}")


if __name__ == "__main__":
    # Beispiel: erst neue Dateien kopieren, dann EC-Lab starten
    SRC_DIR = r"S:\Group Members\Felix_Nagler\Raw-Data_Biologic"
    DEST_DIR = DATA_DIRECTORY

    copied_count = find_and_copy_modified_files(SRC_DIR, DEST_DIR, ".mpr")
    print(f"\n{copied_count} neue Dateien wurden vorbereitet.\n")

    automate_ec_lab()
