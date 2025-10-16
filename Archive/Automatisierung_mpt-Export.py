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