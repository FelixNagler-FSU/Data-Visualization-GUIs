# Importieren aller notwendigen Bibliotheken
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from pathlib import Path
import os
import ast
import matplotlib.font_manager as font_manager
import pickle
from tkinter import Tk, Label, Button, Entry, Text, filedialog, messagebox, Checkbutton, StringVar, BooleanVar, \
    Toplevel, END, DISABLED, NORMAL, scrolledtext, Frame

# Import für die Einbettung von Matplotlib in Tkinter
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Matplotlib-Backend für die GUI einstellen.
# Wichtig: 'TkAgg' muss verwendet werden, da die GUI mit Tkinter erstellt wird.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# --- Hauptlogik für die Datenverarbeitung und das Plotten ---
def process_and_plot(app_instance, data_path_str, dictionary_path_str, different_batches_str, number_of_cells_str,
                     charge_graph_bool, discharge_graph_bool, ce_graph_bool, color_list_str, marker_list_str):
    """
    Verarbeitet die Daten und erstellt die Plots.

    Args:
        app_instance (object): Instanz der GUI-Anwendung zum Loggen von Nachrichten.
        data_path_str (str): Pfad zum Datenverzeichnis.
        dictionary_path_str (str): Pfad zur Dictionary-Datei.
        different_batches_str (str): Anzahl der verschiedenen Batches.
        number_of_cells_str (str): Komma-separierte String der Zellenzahlen pro Batch.
        charge_graph_bool (bool): Flag, ob die Ladekapazität geplottet werden soll.
        discharge_graph_bool (bool): Flag, ob die Entladekapazität geplottet werden soll.
        ce_graph_bool (bool): Flag, ob die Coulombic Efficiency geplottet werden soll.
        color_list_str (str): Komma-separierte String der Farben.
        marker_list_str (str): Komma-separierte String der Marker-Symbole.
    """
    app_instance.log_message("--- Neuer Durchlauf gestartet ---")

    try:
        # Konvertierung der Eingabepfade und Zahlen
        data_path = Path(data_path_str)
        if not data_path.is_dir():
            app_instance.log_message(f"Fehler: Datenverzeichnis nicht gefunden: {data_path_str}")
            messagebox.showerror("Eingabefehler", "Datenverzeichnis nicht gefunden. Bitte prüfen Sie den Pfad.")
            return

        data_file_names = os.listdir(data_path)
        data_file_names.sort()  # Wichtig für konsistente Verarbeitung

        dictionary_name = Path(dictionary_path_str)
        if not dictionary_name.is_file():
            app_instance.log_message(f"Fehler: Dictionary-Datei nicht gefunden: {dictionary_path_str}")
            messagebox.showerror("Eingabefehler", "Dictionary-Datei nicht gefunden. Bitte prüfen Sie den Pfad.")
            return

        different_batches = int(different_batches_str)
        number_of_cells = [int(x.strip()) for x in number_of_cells_str.split(',')]

        # Neue Listen aus den GUI-Eingaben
        color_list = [c.strip() for c in color_list_str.split(',')]
        marker_list = [m.strip() for m in marker_list_str.split(',')]

        # Überprüfung der Eingabekonsistenz
        if sum(number_of_cells) != len(data_file_names):
            app_instance.log_message("Eingabefehler: Gesamt-Zellenzahl stimmt nicht mit Anzahl der Dateien überein.")
            messagebox.showerror("Eingabefehler",
                                 "Die Gesamtzahl der Zellen stimmt nicht mit der Anzahl der Dateien im Verzeichnis überein.")
            return

        if not (charge_graph_bool or discharge_graph_bool or ce_graph_bool):
            app_instance.log_message("Keine Plot-Optionen ausgewählt.")
            messagebox.showwarning("Keine Plots", "Bitte mindestens eine Plot-Option auswählen.")
            return

    except (ValueError, FileNotFoundError) as e:
        app_instance.log_message(f"Fehler bei der Konvertierung der Eingabe oder Dateipfad nicht gefunden: {e}")
        messagebox.showerror("Eingabefehler",
                             f"Fehler bei der Konvertierung der Eingabe oder Dateipfad nicht gefunden: {e}")
        return

    # Dictionaries und Listen initialisieren
    weight_dict = {}
    specific_charge_capacity_list = []
    specific_discharge_capacity_list = []

    # Lese das Dictionary mit den Legendeninformationen aus der Datei
    try:
        with open(dictionary_name, "r") as file:
            contents = file.read()
            dic_legend_list = ast.literal_eval(contents)
    except Exception as e:
        app_instance.log_message(f"Fehler beim Einlesen oder Parsen des Dictionarys: {e}")
        messagebox.showerror("Fehler beim Einlesen", f"Fehler beim Einlesen oder Parsen des Dictionarys: {e}")
        return

    # Berechnet die Zellennummer, bei der der erste Zyklus eines neuen Batches beginnt
    cell_numeration = [sum(number_of_cells[0:counter_var]) for counter_var in range(0, len(number_of_cells))]
    cell_numeration.append(sum(number_of_cells))

    # Einlesen und Verarbeiten der Daten in einer einzigen Schleife
    processed_files_count = 0
    for counter_var in range(0, len(data_file_names)):
        file_path = data_path / data_file_names[counter_var]
        filename = data_file_names[counter_var]
        app_instance.log_message(f"Verarbeite Datei: {filename}")

        header_line_number = None
        file_weight = None

        try:
            with open(file_path, "r", encoding="cp1252") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if "mode	ox/red	error" in line.strip():
                        header_line_number = i - 3
                    if "mass of active material" in line.lower():
                        try:
                            parts = line.split(":")
                            number_str = parts[1].strip().split(' ')[0]
                            number_float = float(number_str.replace(',', '.')) / 1000
                            file_weight = number_float
                            weight_dict[filename] = file_weight
                        except (IndexError, ValueError):
                            app_instance.log_message(f"  - Warnung: Konnte Aktivmasse in {filename} nicht auslesen.")

        except Exception as e:
            app_instance.log_message(f"  - Fehler beim Einlesen der Datei {filename}. Überspringe. Fehler: {e}")
            continue

        if header_line_number is None or file_weight is None:
            app_instance.log_message(f"  - Überspringe Datei: Header oder Aktivmasse in {filename} fehlt.")
            continue

        try:
            data_df = pd.read_table(
                filepath_or_buffer=file_path,
                sep='\t',
                header=header_line_number,
                decimal=',',
                encoding='cp1252'
            )

            data_df.rename(columns={'Q discharge/mA.h': 'DisCap', 'Q charge/mA.h': 'ChCap', 'half cycle': 'Half_cycle'},
                           inplace=True)

            half_cycles = data_df['Half_cycle']
            half_cycles_diff = half_cycles.diff(periods=1)
            cycle_index = half_cycles_diff.loc[half_cycles_diff > 0.5].index - 1

            filtered_DisCap = data_df['DisCap'].loc[cycle_index]
            filtered_ChCap = data_df['ChCap'].loc[cycle_index]

            filtered_DisCap = filtered_DisCap.loc[filtered_DisCap > 0.0] / file_weight
            filtered_ChCap = filtered_ChCap.loc[filtered_ChCap > 0.0] / file_weight

            specific_discharge_capacity_list.append(filtered_DisCap.reset_index(drop=True))
            specific_charge_capacity_list.append(filtered_ChCap.reset_index(drop=True))
            processed_files_count += 1
            app_instance.log_message(f"  - Datei erfolgreich verarbeitet.")

        except Exception as e:
            app_instance.log_message(f"  - Fehler bei der Datenverarbeitung für {filename}: {e}")
            continue

    if not specific_discharge_capacity_list:
        app_instance.log_message("Fehler: Keine gültigen Dateien für die Verarbeitung gefunden.")
        messagebox.showerror("Fehler", "Keine gültigen Dateien für die Verarbeitung gefunden.")
        return

    app_instance.log_message(
        f"Verarbeitung abgeschlossen. {processed_files_count} von {len(data_file_names)} Dateien erfolgreich verarbeitet.")
    app_instance.log_message("Berechne Mittelwerte und Standardabweichungen...")

    # Finde die maximale Länge aller eingelesenen Daten
    max_len = max(len(s) for s in specific_discharge_capacity_list)

    # Auffüllen der Listen mit NaN-Werten, um einheitliche Länge zu gewährleisten
    padded_dis_list = [series.reindex(range(max_len), fill_value=np.nan) for series in specific_discharge_capacity_list]
    padded_ch_list = [series.reindex(range(max_len), fill_value=np.nan) for series in specific_charge_capacity_list]

    # Erstellen der endgültigen DataFrames aus den aufgefüllten Listen
    specific_discharge_capacity = pd.DataFrame({
        f'{data_file_names[i]} DisCap': padded_dis_list[i] for i in range(len(padded_dis_list))
    })

    specific_charge_capacity = pd.DataFrame({
        f'{data_file_names[i]} ChCap': padded_ch_list[i] for i in range(len(padded_ch_list))
    })

    coulombic_efficency = (specific_discharge_capacity / specific_charge_capacity) * 100

    # Berechnung der Mittelwerte und Standardabweichungen pro Batch
    for counter_var in range(0, different_batches):
        batch_cols_dis = specific_discharge_capacity.iloc[:,
                         cell_numeration[counter_var]:cell_numeration[counter_var + 1]]
        specific_discharge_capacity[
            f'{data_file_names[cell_numeration[counter_var]]} mean discharge capacity'] = batch_cols_dis.mean(axis=1)
        specific_discharge_capacity[f'{data_file_names[cell_numeration[counter_var]]} stddev'] = batch_cols_dis.std(
            axis=1)

        batch_cols_ch = specific_charge_capacity.iloc[:, cell_numeration[counter_var]:cell_numeration[counter_var + 1]]
        specific_charge_capacity[
            f'{data_file_names[cell_numeration[counter_var]]} mean charge capacity'] = batch_cols_ch.mean(axis=1)
        specific_charge_capacity[f'{data_file_names[cell_numeration[counter_var]]} stddev'] = batch_cols_ch.std(axis=1)

        batch_cols_ce = coulombic_efficency.iloc[:, cell_numeration[counter_var]:cell_numeration[counter_var + 1]]
        coulombic_efficency[f'{data_file_names[cell_numeration[counter_var]]} mean'] = batch_cols_ce.mean(axis=1)
        coulombic_efficency[f'{data_file_names[cell_numeration[counter_var]]} stddev'] = batch_cols_ce.std(axis=1)

    # Erstellen der Legendenliste
    columns_name_list = list(specific_discharge_capacity.columns.values)
    batch_names_list = [x for x in columns_name_list if 'mean discharge capacity' in x]
    legend_list = []
    for ele in batch_names_list:
        found = False
        for key in dic_legend_list.keys():
            if key in ele:
                legend_list.append(str(dic_legend_list[key][0]))
                found = True
                break
        if not found:
            legend_list.append("Unbekannter Batch")

    if len(legend_list) != different_batches:
        app_instance.log_message(
            'Warnung: Nicht alle Batches wurden im Dictionary gefunden. Die Plot-Legende ist möglicherweise unvollständig.')

    # Hinzufügen der Cycle-Spalte
    specific_discharge_capacity['Cycle'] = np.arange(1, specific_discharge_capacity.shape[0] + 1)
    coulombic_efficency['Cycle'] = np.arange(1, coulombic_efficency.shape[0] + 1)

    # Anpassen der Plot-Grenzen
    max_cycle_dis = specific_discharge_capacity.shape[0]
    max_cycle_ce = coulombic_efficency.shape[0]

    # Plotting der Graphen
    app_instance.log_message("Erstelle Graphen...")

    # Überprüfen, ob die Listen genügend Elemente haben
    num_batches = different_batches
    if len(color_list) < num_batches:
        app_instance.log_message("Warnung: Weniger Farben als Batches. Farben werden wiederverwendet.")
        color_list = (color_list * (num_batches // len(color_list) + 1))[:num_batches]
    if len(marker_list) < num_batches:
        app_instance.log_message("Warnung: Weniger Marker als Batches. Marker werden wiederverwendet.")
        marker_list = (marker_list * (num_batches // len(marker_list) + 1))[:num_batches]

    # Bestimme die Anzahl der Subplots
    num_plots = 0
    if ce_graph_bool:
        num_plots += 1
    if charge_graph_bool or discharge_graph_bool:
        num_plots += 1

    # Erstelle die Subplots basierend auf der Auswahl
    fig, axs = plt.subplots(nrows=num_plots, ncols=1, figsize=(7, 5), squeeze=False)

    ax_index = 0
    ce_ax = None
    capacity_ax = None

    # Plotten des CE-Graphen
    if ce_graph_bool:
        ce_ax = axs[ax_index][0]
        for counter_var in range(0, different_batches):
            ce_ax.errorbar(coulombic_efficency['Cycle'],
                           coulombic_efficency[f'{data_file_names[cell_numeration[counter_var]]} mean'],
                           coulombic_efficency[f'{data_file_names[cell_numeration[counter_var]]} stddev'],
                           capsize=2.5, errorevery=10, marker=marker_list[counter_var],
                           markersize=6, color=color_list[counter_var])
        ce_ax.set_xlim(0, max_cycle_ce + 1)
        ce_ax.set_ylabel('CE [%]')
        ce_ax.set_ylim(98, 102)
        ce_ax.grid()
        ax_index += 1

    # Plotten des Kapazitäts-Graphen
    if charge_graph_bool or discharge_graph_bool:
        capacity_ax = axs[ax_index][0]
        for counter_var in range(0, different_batches):
            label_suffix = ""
            if charge_graph_bool and discharge_graph_bool:
                label_suffix = " (Entladung)"
            elif charge_graph_bool:
                label_suffix = " (Ladung)"

            if discharge_graph_bool:
                capacity_ax.errorbar(specific_discharge_capacity['Cycle'],
                                     specific_discharge_capacity[
                                         f'{data_file_names[cell_numeration[counter_var]]} mean discharge capacity'],
                                     specific_discharge_capacity[
                                         f'{data_file_names[cell_numeration[counter_var]]} stddev'],
                                     label=legend_list[counter_var] + label_suffix, capsize=2.5,
                                     marker=marker_list[counter_var],
                                     color=color_list[counter_var], markersize=6, errorevery=10)

            if charge_graph_bool:
                capacity_ax.errorbar(specific_charge_capacity['Cycle'],
                                     specific_charge_capacity[
                                         f'{data_file_names[cell_numeration[counter_var]]} mean charge capacity'],
                                     specific_charge_capacity[
                                         f'{data_file_names[cell_numeration[counter_var]]} stddev'],
                                     label=legend_list[counter_var] + " (Ladung)", capsize=2.5,
                                     marker=marker_list[counter_var],
                                     color=color_list[counter_var], markersize=6, linestyle='--', errorevery=10)

        capacity_ax.set_xlabel('Cycle')
        capacity_ax.set_ylabel('Kapazität [mAh $\\ g^{-1}$]')
        capacity_ax.legend(fontsize=12, loc=0)
        capacity_ax.set_xlim(0, max_cycle_dis + 1)
        capacity_ax.set_ylim(bottom=0)
        capacity_ax.grid()

    # Anpassen des Layouts und Speichern der Achsen
    if num_plots == 2:
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        axs[0][0].set_xticklabels([])
        fig.suptitle('Kapazitäts- und CE-Verlauf')
    elif ce_graph_bool:
        plt.tight_layout()
        fig.suptitle('CE-Verlauf')
    elif charge_graph_bool or discharge_graph_bool:
        plt.tight_layout()
        fig.suptitle('Kapazitäts-Verlauf')

    app_instance.current_axes = [ce_ax, capacity_ax] if ce_graph_bool and (
                charge_graph_bool or discharge_graph_bool) else [ce_ax] if ce_graph_bool else [capacity_ax]

    # Speichern der Graphen und Daten
    app_instance.log_message("Graph erfolgreich erstellt. Versuche, die Daten zu speichern...")
    try:
        # Erstelle den Ordner, falls er nicht existiert
        save_dir = data_path.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(os.path.join(str(save_dir), data_path.parts[-2]))
        plt.savefig(os.path.join(str(save_dir), str(str(data_path.name) + '_' + str(data_path.parts[-2]) + '.svg')))
        plt.savefig(os.path.join(str(save_dir), str(str(data_path.name) + '_' + str(data_path.parts[-2]))))

        # Speichern des gesamten Matplotlib-Figure-Objekts für die spätere Bearbeitung
        pickle_file_path = os.path.join(str(save_dir), data_path.parts[-2] + '_plot_data.pickle')
        pickle.dump(fig, open(pickle_file_path, 'wb'))
        app_instance.log_message(f"Plot-Objekt als Pickle-Datei gespeichert: {pickle_file_path}")

    except Exception as e:
        app_instance.log_message(f"Warnung: Das Speichern der Plot-Dateien ist fehlgeschlagen. Fehler: {e}")

    # Speichern der Daten in eine .txt-Datei
    save_file_name = os.path.join(str(data_path.parent),
                                  str(str(data_path.name) + '_' + str(data_path.parts[-2]) + '.txt'))
    df_to_save = specific_discharge_capacity.copy()
    if ce_graph_bool:
        ce_cols_to_add = [col for col in coulombic_efficency.columns if 'mean' in col or 'stddev' in col]
        for col in ce_cols_to_add:
            df_to_save[col + ' CE'] = coulombic_efficency[col]

    try:
        df_to_save.to_csv(save_file_name, sep=',', index=False)
        app_instance.log_message(f"Daten wurden in {save_file_name} gespeichert.")
    except Exception as e:
        app_instance.log_message(f"Warnung: Das Speichern der CSV-Datei ist fehlgeschlagen. Fehler: {e}")

    # Entferne den alten Plot und erstelle einen neuen Canvas im GUI
    app_instance.plot_frame.pack_forget()
    app_instance.plot_frame = Frame(app_instance.root)
    app_instance.plot_frame.grid(row=8, column=0, columnspan=4, padx=10, pady=5, sticky="nsew")

    canvas = FigureCanvasTkAgg(fig, master=app_instance.plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

    toolbar = NavigationToolbar2Tk(canvas, app_instance.plot_frame)
    toolbar.update()
    canvas.get_tk_widget().pack(side='top', fill='both', expand=1)

    app_instance.current_figure = fig
    app_instance.edit_plot_button.config(state=NORMAL)
    app_instance.log_message("Plot-Objekt wurde in der GUI angezeigt.")
    plt.close(fig)  # Schließt die redundante Matplotlib-Fenster


# --- GUI-Setup mit Tkinter ---
class EchemPlotterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Electrochemical Data Plotter")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_columnconfigure(3, weight=1)

        # Variablen für die Pfade
        self.data_path = StringVar()
        self.dictionary_path = StringVar()
        self.different_batches_var = StringVar()
        self.number_of_cells_var = StringVar()
        self.ce_graph_var = BooleanVar()
        self.charge_graph_var = BooleanVar()
        self.discharge_graph_var = BooleanVar(value=True)  # Entladekapazität ist standardmäßig ausgewählt
        self.color_list_var = StringVar(
            value='tab:blue, tab:orange, tab:green, tab:red, tab:purple, tab:brown, tab:pink, tab:gray, tab:olive, tab:cyan')
        self.marker_list_var = StringVar(value='o, v, ^, <, >, s, p, 2, 3, 4, 8, s, p, P, *, h, H, +, x, X, D, d, |, _')

        self.current_figure = None
        self.current_axes = None

        self.setup_ui()

    def setup_ui(self):
        # Eingabebereich für Datenpfad
        Label(self.root, text="Datenverzeichnis auswählen:").grid(row=0, column=0, sticky='w', padx=10, pady=5)
        Button(self.root, text="Durchsuchen...", command=self.select_data_folder).grid(row=0, column=1, padx=10, pady=5)
        Label(self.root, textvariable=self.data_path, width=40).grid(row=0, column=2, padx=10, pady=5)

        # Eingabebereich für Dictionary-Datei
        Label(self.root, text="Dictionary-Datei auswählen:").grid(row=1, column=0, sticky='w', padx=10, pady=5)
        Button(self.root, text="Durchsuchen...", command=self.select_dictionary_file).grid(row=1, column=1, padx=10,
                                                                                           pady=5)
        Label(self.root, textvariable=self.dictionary_path, width=40).grid(row=1, column=2, padx=10, pady=5)

        # Neuer Button für das Bearbeiten des Dictionarys
        self.edit_dict_button = Button(self.root, text="Dictionary anzeigen & bearbeiten",
                                       command=self.open_dictionary_editor, state=DISABLED)
        self.edit_dict_button.grid(row=1, column=3, padx=10, pady=5)

        # Eingabebereich für Anzahl Batches
        Label(self.root, text="Anzahl verschiedener Batches:").grid(row=2, column=0, sticky='w', padx=10, pady=5)
        self.batches_entry = Entry(self.root, textvariable=self.different_batches_var)
        self.batches_entry.grid(row=2, column=1, columnspan=2, sticky='we', padx=10, pady=5)

        # Eingabebereich für Anzahl der Zellen pro Batch
        Label(self.root, text="Anzahl der Zellen pro Batch (z.B. 2,1,3):").grid(row=3, column=0, sticky='w', padx=10,
                                                                                pady=5)
        self.cells_entry = Entry(self.root, textvariable=self.number_of_cells_var)
        self.cells_entry.grid(row=3, column=1, columnspan=2, sticky='we', padx=10, pady=5)

        # Eingabebereich für Farben und Marker
        Label(self.root, text="Farben (kommasep.):").grid(row=4, column=0, sticky='w', padx=10, pady=5)
        self.colors_entry = Entry(self.root, textvariable=self.color_list_var)
        self.colors_entry.grid(row=4, column=1, columnspan=3, sticky='we', padx=10, pady=5)
        Label(self.root, text="Marker (kommasep.):").grid(row=5, column=0, sticky='w', padx=10, pady=5)
        self.markers_entry = Entry(self.root, textvariable=self.marker_list_var)
        self.markers_entry.grid(row=5, column=1, columnspan=3, sticky='we', padx=10, pady=5)

        # Checkboxen für die Plot-Optionen
        Label(self.root, text="Graphen-Typen:").grid(row=6, column=0, sticky='w', padx=10, pady=5)
        Checkbutton(self.root, text="Entladekapazität anzeigen", variable=self.discharge_graph_var).grid(row=6,
                                                                                                         column=1,
                                                                                                         sticky='w',
                                                                                                         padx=10,
                                                                                                         pady=5)
        Checkbutton(self.root, text="Ladekapazität anzeigen", variable=self.charge_graph_var).grid(row=6, column=2,
                                                                                                   sticky='w', padx=10,
                                                                                                   pady=5)
        Checkbutton(self.root, text="CE-Diagramm anzeigen", variable=self.ce_graph_var).grid(row=6, column=3,
                                                                                             sticky='w', padx=10,
                                                                                             pady=5)

        # Plot-Buttons
        Button(self.root, text="Start Plotting", command=self.run_plotting).grid(row=7, column=0, columnspan=2, pady=10)
        self.edit_plot_button = Button(self.root, text="Plot-Eigenschaften bearbeiten", command=self.open_plot_editor,
                                       state=DISABLED)
        self.edit_plot_button.grid(row=7, column=2, columnspan=2, pady=10)

        # Frame für den Matplotlib-Plot
        self.plot_frame = Frame(self.root, bg="white", borderwidth=2, relief="groove")
        self.plot_frame.grid(row=8, column=0, columnspan=4, padx=10, pady=5, sticky="nsew")
        self.root.grid_rowconfigure(8, weight=1)

        # Log-Bereich
        Label(self.root, text="Debugging-Protokoll:").grid(row=9, column=0, sticky='w', padx=10, pady=5)
        self.log_text = scrolledtext.ScrolledText(self.root, height=10, width=80)
        self.log_text.grid(row=10, column=0, columnspan=4, padx=10, pady=5, sticky='nsew')
        self.root.grid_rowconfigure(10, weight=1)

    def select_data_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.data_path.set(folder_selected)

    def select_dictionary_file(self):
        file_selected = filedialog.askopenfilename(defaultextension=".txt",
                                                   filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if file_selected:
            self.dictionary_path.set(file_selected)
            self.edit_dict_button.config(state=NORMAL)  # Aktiviert den Bearbeiten-Button

    def log_message(self, message):
        """Fügt eine Nachricht zum Textfeld hinzu."""
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)  # Scrollt automatisch zum Ende

    def open_dictionary_editor(self):
        """Öffnet ein neues Fenster zum Bearbeiten des Dictionarys."""
        dict_file_path = self.dictionary_path.get()
        if not dict_file_path:
            messagebox.showwarning("Fehlende Datei", "Bitte zuerst eine Dictionary-Datei auswählen.")
            return

        editor_window = Toplevel(self.root)
        editor_window.title("Dictionary-Editor")

        # Lade den Inhalt der Datei
        try:
            with open(dict_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte Datei nicht laden: {e}")
            editor_window.destroy()
            return

        Label(editor_window, text=f"Bearbeiten von: {os.path.basename(dict_file_path)}").pack(pady=5)
        editor_text = Text(editor_window, width=80, height=20)
        editor_text.pack(padx=10, pady=5)
        editor_text.insert(END, content)

        def save_and_close():
            """Speichert den bearbeiteten Inhalt und schließt das Fenster."""
            new_content = editor_text.get("1.0", END)
            try:
                # Prüfen, ob der Inhalt ein gültiges Dictionary ist
                ast.literal_eval(new_content)
                with open(dict_file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                self.log_message("Dictionary erfolgreich gespeichert.")
                editor_window.destroy()
            except (ValueError, SyntaxError) as e:
                messagebox.showerror("Speicherfehler",
                                     f"Ungültiges Dictionary-Format. Bitte korrigieren Sie es. Fehler: {e}")

        save_button = Button(editor_window, text="Speichern & Schließen", command=save_and_close)
        save_button.pack(pady=5)

        cancel_button = Button(editor_window, text="Abbrechen", command=editor_window.destroy)
        cancel_button.pack(pady=5)

    def open_plot_editor(self):
        """Öffnet ein neues Fenster zum Bearbeiten der Plot-Eigenschaften."""
        if self.current_figure is None:
            messagebox.showwarning("Kein Plot geladen", "Bitte zuerst einen Plot erstellen oder laden.")
            return

        editor_window = Toplevel(self.root)
        editor_window.title("Plot-Eigenschaften bearbeiten")

        # Variablen für die Eingabefelder
        title_var = StringVar(value=self.current_figure.get_suptitle())
        xlabel_var = StringVar(value=self.current_axes[-1].get_xlabel())
        ylabel_var = StringVar(value=self.current_axes[-1].get_ylabel())

        # Behandelt den Fall mit zwei Subplots für CE
        if len(self.current_axes) > 1:
            ce_ylabel_var = StringVar(value=self.current_axes[0].get_ylabel())

        xmin_var = StringVar(value=str(self.current_axes[-1].get_xlim()[0]))
        xmax_var = StringVar(value=str(self.current_axes[-1].get_xlim()[1]))
        ymin_var = StringVar(value=str(self.current_axes[-1].get_ylim()[0]))
        ymax_var = StringVar(value=str(self.current_axes[-1].get_ylim()[1]))

        # UI-Elemente
        Label(editor_window, text="Titel:").grid(row=0, column=0, padx=5, pady=2, sticky='e')
        Entry(editor_window, textvariable=title_var).grid(row=0, column=1, padx=5, pady=2, sticky='we')

        Label(editor_window, text="X-Achse Label:").grid(row=1, column=0, padx=5, pady=2, sticky='e')
        Entry(editor_window, textvariable=xlabel_var).grid(row=1, column=1, padx=5, pady=2, sticky='we')

        Label(editor_window, text="Y-Achse Label:").grid(row=2, column=0, padx=5, pady=2, sticky='e')
        Entry(editor_window, textvariable=ylabel_var).grid(row=2, column=1, padx=5, pady=2, sticky='we')

        if len(self.current_axes) > 1:
            Label(editor_window, text="CE-Achse Y-Label:").grid(row=3, column=0, padx=5, pady=2, sticky='e')
            Entry(editor_window, textvariable=ce_ylabel_var).grid(row=3, column=1, padx=5, pady=2, sticky='we')

        Label(editor_window, text="X-Achse Limits (Min, Max):").grid(row=4, column=0, padx=5, pady=2, sticky='e')
        Entry(editor_window, textvariable=xmin_var).grid(row=4, column=1, padx=5, pady=2, sticky='we')
        Entry(editor_window, textvariable=xmax_var).grid(row=4, column=2, padx=5, pady=2, sticky='we')

        Label(editor_window, text="Y-Achse Limits (Min, Max):").grid(row=5, column=0, padx=5, pady=2, sticky='e')
        Entry(editor_window, textvariable=ymin_var).grid(row=5, column=1, padx=5, pady=2, sticky='we')
        Entry(editor_window, textvariable=ymax_var).grid(row=5, column=2, padx=5, pady=2, sticky='we')

        def apply_changes():
            try:
                # Anwenden der Änderungen
                self.current_figure.suptitle(title_var.get())
                self.current_axes[-1].set_xlabel(xlabel_var.get())
                self.current_axes[-1].set_ylabel(ylabel_var.get())

                if len(self.current_axes) > 1:
                    self.current_axes[0].set_ylabel(ce_ylabel_var.get())

                new_xmin = float(xmin_var.get())
                new_xmax = float(xmax_var.get())
                new_ymin = float(ymin_var.get())
                new_ymax = float(ymax_var.get())

                for ax in self.current_axes:
                    ax.set_xlim(new_xmin, new_xmax)
                    ax.set_ylim(new_ymin, new_ymax)

                self.current_figure.canvas.draw()
                self.log_message("Plot-Eigenschaften erfolgreich aktualisiert.")
                editor_window.destroy()

            except ValueError:
                messagebox.showerror("Eingabefehler", "Ungültiges Format für Achsenlimits. Bitte nur Zahlen verwenden.")

        Button(editor_window, text="Übernehmen", command=apply_changes).grid(row=6, column=0, columnspan=3, pady=10)

    def run_plotting(self):
        data_path = self.data_path.get()
        dictionary_path = self.dictionary_path.get()
        different_batches = self.different_batches_var.get()
        number_of_cells = self.number_of_cells_var.get()
        charge_graph = self.charge_graph_var.get()
        discharge_graph = self.discharge_graph_var.get()
        ce_graph = self.ce_graph_var.get()
        color_list_str = self.color_list_var.get()
        marker_list_str = self.marker_list_var.get()

        if not all([data_path, dictionary_path, different_batches, number_of_cells]):
            messagebox.showwarning("Fehlende Eingabe", "Bitte alle Felder ausfüllen.")
            return

        # Die Prozess- und Plot-Funktion aufrufen
        process_and_plot(self, data_path, dictionary_path, different_batches, number_of_cells, charge_graph,
                         discharge_graph, ce_graph, color_list_str, marker_list_str)


if __name__ == "__main__":
    root = Tk()
    app = EchemPlotterApp(root)
    root.mainloop()
