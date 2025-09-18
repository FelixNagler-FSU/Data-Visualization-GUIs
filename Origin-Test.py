from pywinauto import Application
import time

# Schritt 1: Verbindung zum EC-Lab-Fenster herstellen
# WICHTIG: Passen Sie den Fenstertitel an den genauen Titel auf Ihrem System an.
# Nach dem Klick auf "Experiment" kann sich der Titel ändern.
window_title = "EC-Lab V11.62.9 - [VMP-300 - virtual, channel 1 - no experiment]"
app = Application(backend="uia").connect(title=window_title)

# Eine Referenz auf das Hauptfenster erstellen.
ec_lab_main_window = app.window(title=window_title)

# Experiment-Menü suchen und anklicken.
print("Klicke auf 'Experiment' Menü...")
try:
    # WICHTIG: Suchen Sie das Element nur anhand des Titels, der Steuerungsart oder der ID.
    # Vermeiden Sie die Verwendung von generierten Namen wie "ECLabVVMPVirtualChannelNoExperiment".
    Experiment = ec_lab_main_window.child_window(title="Experiment", control_type="MenuItem").wrapper_object()
    Experiment.click_input()
except Exception as e:
    print(f"Fehler beim Klicken auf das 'Experiment'-Menü: {e}")
    # Beenden Sie das Skript, wenn das Menü nicht gefunden wird.
    exit()

# Geben Sie dem Programm Zeit, um das Menü zu öffnen.
time.sleep(2)

# Schritt 2: Die Verbindung zum Fenster nach dem Klick erneuern.
# Dies ist der entscheidende Schritt, um den aktuellen Titel zu erhalten.
print("Aktualisiere die Fenster-Referenz und rufe den neuen Titel ab...")
try:
    # Erneut mit dem Fenster verbinden. Dies stellt sicher, dass alle
    # UI-Elemente und der Titel auf dem neuesten Stand sind.
    # Sie können nach dem ursprünglichen Titel oder einem Teil des Titels suchen.
    updated_app = Application(backend="uia").connect(title_re=".*EC-Lab.*")
    updated_window = updated_app.top_window()

    # Geben Sie die aktualisierten Steuerelemente aus.
    updated_window.print_control_identifiers()

    # Geben Sie den aktuellen Fenstertitel aus.
    current_title = updated_window.window_text()
    print(f"\nAktueller Fenstertitel: {current_title}")

except Exception as e:
    print(f"Fehler beim erneuten Verbinden mit dem Fenster: {e}")
