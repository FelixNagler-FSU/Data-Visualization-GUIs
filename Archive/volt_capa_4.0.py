# Import all necessary libraries
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from pathlib import Path
import os
import ast
import matplotlib.font_manager as font_manager
import pickle

#########################################################################
# Einstellungen
# Configuration
data_path = Path(r'C:\Users\ro45vij\Desktop\AA_Data-Processing\AA_Plotting\E4a\data')
data_file_names = os.listdir(data_path)

# ist erster Zyklus ein Discharge
first_cycle_discharge_only = 'yes'
#first_cycle_discharge_only = 'no'

# Korrektur: CycleNumber als Liste oder Tupel definieren
# Correction: Define CycleNumber as a list or tuple
CycleNumber = 1  # Eingabe der Zyklen die geplottet werden sollen
Inlet = 0  # Wenn Inlet dann ==1, kein Inlet wenn !=1

# C-Rate von Zyklus, taucht als Überschrift in Graph auf
# C-Rate of the cycle, appears as a heading in the graph
CRate = 'C/10'

# Name des Dictionaries
# Name of the dictionary
dictionary_name = 'dictionary_HIPOLE.txt'

# Einstellen Schriftart
# Set font
# plt.rcParams.update({'font.family':'Times New Roman'})
# plt.rcParams.update({'font.size': 10})
# plt.rcParams.update({'mathtext.default': 'regular' })

###################################################################

# Aus CycleNumber Eingabe wird Liste gemacht
# The CycleNumber input is converted into a list
if not isinstance(CycleNumber, (list, tuple)):
    CycleNumber = [CycleNumber]

# Initialisierung der Listen und Dictionaries
# Initialization of lists and dictionaries
max_rows = 0
all_data = {cycle: {} for cycle in CycleNumber}
# Neues Dictionary zum Speichern der Aktivmassen pro Datei
# New dictionary to store the active masses per file
weight_dict = {}

# Einlesen der Daten aus angegebenen Verzeichnis
# Reading the data from the specified directory
for idx, filename in enumerate(data_file_names):
    file_path = data_path / filename

    # Dynamisches Auslesen der tatsächlichen Header-Zeile durch Suche nach dem Keyword
    # Dynamically read the actual header row by searching for the keyword
    header_line_number = None
    try:
        with open(file_path, "r", encoding="cp1252") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # Die Zeilennummer ist i - 3, da Python's enumerate 0-basiert ist aber Leerzeilen nicht einließt
                # The line number is i - 3 because Python's enumerate is 0-indexed but do not read in empty lines
                if "mode	ox/red	error" in line.strip():
                    header_line_number = i - 3
                # Header-Information für Aktivmaterialgewicht auslesen
                # Read header information for active material weight
                if "mass of active material" in line.lower():
                    try:
                        parts = line.split(":")
                        number_str = parts[1].strip().split(' ')[0]
                        # Komma durch Punkt ersetzen und in Kilogramm konvertieren (dividing by 1000)
                        # Replace comma with period and convert to kilograms (dividing by 1000)
                        number_float = float(number_str.replace(',', '.')) / 1000
                        weight_dict[filename] = number_float
                        print(f"Aktivmasse für {filename}: {number_float * 1000} mg")
                    except (IndexError, ValueError):
                        print(f"Warnung: Konnte Aktivmasse in {filename} nicht auslesen. Überspringe Datei.")
                        break

    except (IOError, IndexError, ValueError):
        print(f"Warnung: Konnte Header-Zeile in {filename} nicht finden. Überspringe Datei.")
        continue

    # Check if header was found
    if header_line_number is None:
        print(f"Warnung: Header-Zeile 'mode ox/red error' in {filename} nicht gefunden. Überspringe Datei.")
        continue

    # Überprüfen, ob Aktivmasse gefunden wurde
    # Check if active mass was found
    if filename not in weight_dict:
        print(f"Warnung: Keine Aktivmasse in Header für {filename} gefunden. Datei wird übersprungen.")
        continue
    # --- END OF NEW LOGIC ---

    # Lese die Hauptdaten der Datei mit der dynamisch ermittelten Header-Zeile
    # Read the main data of the file with the dynamically determined header row
    df = pd.read_table(
        filepath_or_buffer=file_path,
        sep='\t',
        # Korrektur: Die korrekte Headerzeile verwenden, -1 weil pandas 0-basiert ist
        # Correction: Use the correct header row, -1 because pandas is 0-indexed
        header=header_line_number,
        decimal=',',
        encoding='cp1252'
    )

    # Spaltennamen bereinigen und umbenennen
    # Clean up and rename column names
    df.columns = df.columns.str.strip()

    # Erkenne und benenne die Spalten dynamisch
    # Recognize and rename the columns dynamically
    try:
        # Dynamisches Finden der Ecell-Spalte
        # Dynamically find the Ecell column
        potential_ecell_cols = [col for col in df.columns if 'Ecell/V' in col or 'Ewe/V' in col]
        if not potential_ecell_cols:
            raise KeyError("Ecell-Spalte nicht gefunden")
        df.rename(columns={potential_ecell_cols[0]: 'Ecell'}, inplace=True)

        # Dynamisches Finden der Capacity-Spalte
        # Dynamically find the Capacity column
        potential_capacity_cols = [col for col in df.columns if 'Capacity/mA.h' in col]
        if not potential_capacity_cols:
            raise KeyError("Capacity-Spalte nicht gefunden")
        df.rename(columns={potential_capacity_cols[0]: 'Capacity'}, inplace=True)

        # Dynamisches Finden der half cycle-Spalte
        # Dynamically find the half cycle column
        potential_half_cycle_cols = [col for col in df.columns if 'half cycle' in col]
        if not potential_half_cycle_cols:
            raise KeyError("half cycle-Spalte nicht gefunden")
        df.rename(columns={potential_half_cycle_cols[0]: 'Half_cycle'}, inplace=True)

    except KeyError as e:
        print(f"Fehler beim Finden der Spalten in {filename}: {e}. Überspringe Datei.")
        continue
    # Auskommentiert, da dieser Filter potenziell zu Datenverlust führen kann
    # Commented out, as this filter can potentially lead to data loss
    # if 'Ns' in df.columns:
    #     first_ns_value = df['Ns'].iloc[0]
    #     start_index = (df['Ns'] != first_ns_value).idxmax()
    #     df = df.loc[start_index:].reset_index(drop=True)

    # Bestimme die maximale Zeilenzahl für die Dictionaries
    # Determine the maximum number of rows for the dictionaries
    if df.shape[0] > max_rows:
        max_rows = df.shape[0]

    # Augteilen/Gruppieren der Daten nach half cycles
    # Split/group the data by half cycles
    grouped = df.groupby('Half_cycle')

    # NEU: Ein temporäres Dictionary zum Sammeln der Series
    # NEW: A temporary dictionary to collect the Series
    temp_data_dict = {}

    # Debugging-Ausgabe: Zeigt die Anzahl der Datenpunkte pro Half_cycle
    # Debug output: Shows the number of data points per Half_cycle
    print(f"Verarbeitung von Datei: {filename}")
    for half_cycle_value, group in grouped:
        print(f"  Half_cycle {half_cycle_value}: {len(group)} Datenpunkte")

        # Group data by 'Half_cycle' to separate charge and discharge phases
        grouped = df.groupby('Half_cycle')
        temp_data_dict = {}

        for half_cycle_value, group in grouped:
            if not group.empty:

                # Logic to determine charge/discharge and assign cycle number
                is_charge = False  # Default to discharge
                if first_cycle_discharge_only == 'yes':
                    if half_cycle_value == 0:
                        cycle_num = 1
                    else:
                        cycle_num = (half_cycle_value - 2) // 2 + 2
                        is_charge = (half_cycle_value % 2 == 0)
                else:
                    cycle_num = half_cycle_value // 2 + 1
                    is_charge = (half_cycle_value % 2 == 0)

                # Assign column names based on cycle type (charge/discharge)
                if is_charge:
                    volt_col = f'chVolt{cycle_num}'
                    cap_col = f'chCap{cycle_num}'
                else:
                    volt_col = f'disVolt{cycle_num}'
                    cap_col = f'disCap{cycle_num}'

            # NEU: Daten werden in ein Dictionary geschrieben, nicht direkt in ein DataFrame
            # NEW: Data is written to a dictionary, not directly to a DataFrame
            temp_data_dict[volt_col] = group['Ecell'].reset_index(drop=True)
            temp_data_dict[cap_col] = group['Capacity'].reset_index(drop=True)

    # NEU: DataFrame wird erst am Ende aus dem Dictionary erstellt, um unterschiedliche Längen zu ermöglichen
    # NEW: DataFrame is created from the dictionary at the end to allow for different lengths
    cycle_separated_df = pd.DataFrame(temp_data_dict)

    # Reduzierung der Daten auf die in CycleNumber eingebenen Zyklen
    # Reduction of data to the cycles entered in CycleNumber
    for cycle in CycleNumber:
        chCap_col = f'chCap{cycle}'
        chVolt_col = f'chVolt{cycle}'
        disCap_col = f'disCap{cycle}'
        disVolt_col = f'disVolt{cycle}'

        # Abruf des Gewichts aus dem Dictionary
        # Retrieve the weight from the dictionary
        weight = weight_dict.get(filename, 1)  # Fallback to 1 if weight is not found

        # Normierung der Capacity auf das ausgelesenen Aktivematerial Gewicht
        # Normalization of capacity to the read active material weight
        # Speicherung der Daten in temporärem dicitionary all_data
        # Storage of data in temporary dictionary all_data
        if chCap_col in cycle_separated_df.columns and chVolt_col in cycle_separated_df.columns:
            all_data[cycle][f'{filename}_ch_cap'] = cycle_separated_df[chCap_col] / weight
            all_data[cycle][f'{filename}_ch_volt'] = cycle_separated_df[chVolt_col]
        else:
            print(f"{filename}: Lade-Daten für Zyklus {cycle} nicht gefunden.")

        if disCap_col in cycle_separated_df.columns and disVolt_col in cycle_separated_df.columns:
            all_data[cycle][f'{filename}_dis_cap'] = cycle_separated_df[disCap_col] / weight
            all_data[cycle][f'{filename}_dis_volt'] = cycle_separated_df[disVolt_col]
        else:
            print(f"{filename}: Entlade-Daten für Zyklus {cycle} nicht gefunden.")

# Erstellung der endgültigen DataFrames aus den gesammelten Daten
# Creation of the final DataFrames from the collected data
capacity_charge_all = {cycle: pd.DataFrame({k: v for k, v in all_data[cycle].items() if '_ch_cap' in k}) for cycle in
                       CycleNumber}
voltage_charge_all = {cycle: pd.DataFrame({k: v for k, v in all_data[cycle].items() if '_ch_volt' in k}) for cycle in
                      CycleNumber}
capacity_discharge_all = {cycle: pd.DataFrame({k: v for k, v in all_data[cycle].items() if '_dis_cap' in k}) for cycle
                          in CycleNumber}
voltage_discharge_all = {cycle: pd.DataFrame({k: v for k, v in all_data[cycle].items() if '_dis_volt' in k}) for cycle
                         in CycleNumber}

# Ersetze 0 durch NaN in capacity_charge_all, sonst möglicherweise hässliche Plots
# Replace 0 by NaN in capacity_charge_all, otherwise plots might look bad
for cycle in capacity_charge_all:
    capacity_charge_all[cycle].replace(0, np.nan, inplace=True)

# Dictionary mit Infos über Batches -> für Legenden-Beschriftung
# Dictionary with information about batches -> for legend labels
file = open(dictionary_name, "r")
contents = file.read()
dic_legend_list = ast.literal_eval(contents)
file.close()

# Vergleicht Einträge aus batch_name_list nach einander mit allen keys aus dictionary und erzeugt Legenden Liste
# Compares entries from batch_name_list one after the other with all keys from the dictionary and generates a legend list
legend_list = []
# Hier wird nun über alle Dateien iteriert, für die auch ein Gewicht in der Excel-Datei gefunden wurde
# Now iterate over all files for which a weight was also found in the Excel file
filtered_data_file_names = [f for f in data_file_names if f in weight_dict]
for filename in filtered_data_file_names:
    if filename in weight_dict:
        found = False
        for key in dic_legend_list.keys():
            if key in filename:
                baustein1 = str(dic_legend_list[key][0])
                baustein2 = str(dic_legend_list[key][1])
                baustein3 = str(dic_legend_list[key][2])
                baustein4 = str(dic_legend_list[key][3] + '] mAh/$\\ cm^{2}$')
                legend_list.append('{}'.format(baustein1))
                found = True
                break
        if not found:
            print('Not all batches were found in dictionary')
            import sys

            sys.exit()

# Erzeugung eines Plotes
# Creation of a plot
fig, ax1 = plt.subplots(nrows=1, ncols=1)

# Definition einer Farbliste, sonst haben Charge und Discharge der gleichen Zelle nicht gleiche Farbe
# Definition of a color list, otherwise charge and discharge of the same cell do not have the same color
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan',
              'black', 'indianred', 'lightgreen', 'orchid', 'gold', 'khaki', 'tab:pink', 'tab:gray', 'tab:olive',
              'tab:cyan',
              'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan',
              'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']
linestyle_list = ['--', '-', '--', '-']
alpha_list = [1, 0.2, 0.8, 0.6, 0.4, 0.2]

# Wir benötigen eine neue, gefilterte Liste von Dateinamen für die Plot-Schleife
# We need a new, filtered list of filenames for the plotting loop
filtered_data_file_names = [f for f in data_file_names if f in weight_dict]

for i, cycle in enumerate(CycleNumber):
    for col_idx, filename in enumerate(filtered_data_file_names):
        # Spaltennamen im neuen DataFrame anpassen, um die richtige Spalte zu finden
        # Adjust column names in the new DataFrame to find the correct column
        ch_cap_col = f'{filename}_ch_cap'
        ch_volt_col = f'{filename}_ch_volt'
        dis_cap_col = f'{filename}_dis_cap'
        dis_volt_col = f'{filename}_dis_volt'

        if ch_cap_col in capacity_charge_all[cycle].columns:
            ax1.plot(
                capacity_charge_all[cycle][ch_cap_col],
                voltage_charge_all[cycle][ch_volt_col],
                color=color_list[col_idx],
                alpha=alpha_list[i],
                label=legend_list[col_idx] + '\u00A0' + '(Cycle {cycle})'.format(cycle=cycle)
            )
        if dis_cap_col in capacity_discharge_all[cycle].columns:
            ax1.plot(
                capacity_discharge_all[cycle][dis_cap_col],
                voltage_discharge_all[cycle][dis_volt_col],
                color=color_list[col_idx],
                alpha=alpha_list[i],
            )

ax1.set_xlabel('Capacity [mAh $\\ g^{-1}$]')
ax1.set_ylabel('Voltage [V]')
ax1.legend(fontsize=10)
ax1.grid()
#ax1.set_title(f'Cycles {CycleNumber} ({CRate})')

plt.show()

# Speichern der Graphen
plot_filename_base = str(CycleNumber) + str(data_path.parts[-2])
plot_path = os.path.join(str(data_path.parent), plot_filename_base)

plt.savefig(f'{plot_path}.png')
plt.savefig(f'{plot_path}.svg')
plt.savefig(f'{plot_path}.pdf')
pickle.dump(fig, open(f'{plot_path}.pickle', 'wb'))

# NEU: Speichern der geplotteten Daten in einer Textdatei mit dem gewünschten Spaltenformat
# NEW: Save the plotted data to a text file with the desired column format
with open(f'{plot_path}.txt', 'w') as f:
    # Dynamisch die Kopfzeile erstellen
    header_line = ""
    for cycle in CycleNumber:
        for filename in filtered_data_file_names:
            file_short_name = Path(filename).stem
            header_line += f"ch_cap_C{cycle}_{file_short_name}\tch_volt_C{cycle}_{file_short_name}\t"
            header_line += f"dis_cap_C{cycle}_{file_short_name}\tdis_volt_C{cycle}_{file_short_name}\t"

    f.write(header_line.strip() + '\n')

    # Finde die maximale Anzahl von Zeilen in allen DataFrames
    max_len = 0
    for cycle in CycleNumber:
        for filename in filtered_data_file_names:
            ch_cap_col = f'{filename}_ch_cap'
            if ch_cap_col in capacity_charge_all[cycle].columns:
                max_len = max(max_len, len(capacity_charge_all[cycle][ch_cap_col]))
            dis_cap_col = f'{filename}_dis_cap'
            if dis_cap_col in capacity_discharge_all[cycle].columns:
                max_len = max(max_len, len(capacity_discharge_all[cycle][dis_cap_col]))

    # Schreibe die Daten Zeile für Zeile
    for i in range(max_len):
        row_str = ""
        for cycle in CycleNumber:
            for filename in filtered_data_file_names:
                # Lade-Daten
                ch_cap_col = f'{filename}_ch_cap'
                ch_volt_col = f'{filename}_ch_volt'

                if ch_cap_col in capacity_charge_all[cycle].columns and i < len(capacity_charge_all[cycle][ch_cap_col]):
                    ch_cap_val = capacity_charge_all[cycle][ch_cap_col].iloc[i]
                    ch_volt_val = voltage_charge_all[cycle][ch_volt_col].iloc[i]
                    row_str += f"{ch_cap_val}\t{ch_volt_val}\t"
                else:
                    row_str += "\t\t"  # Zwei leere Spalten

                # Entlade-Daten
                dis_cap_col = f'{filename}_dis_cap'
                dis_volt_col = f'{filename}_dis_volt'
                if dis_cap_col in capacity_discharge_all[cycle].columns and i < len(
                        capacity_discharge_all[cycle][dis_cap_col]):
                    dis_cap_val = capacity_discharge_all[cycle][dis_cap_col].iloc[i]
                    dis_volt_val = voltage_discharge_all[cycle][dis_volt_col].iloc[i]
                    row_str += f"{dis_cap_val}\t{dis_volt_val}\t"
                else:
                    row_str += "\t\t"  # Zwei leere Spalten

        f.write(row_str.strip() + '\n')

print(f"Plot-Daten wurden in {plot_path}.txt gespeichert.")