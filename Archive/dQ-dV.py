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
data_path = Path(r'C:\Users\ro45vij\Documents\data_1')
data_file_names = os.listdir(data_path)


CycleNumber = 5   # Eingabe der Zyklen die geplottet werden sollen
Inlet = 0           # Wenn Inlet dann ==1, kein Inlet wenn !=1

# C-Rate von Zyklus, taucht als Überschrift in Graph auf
CRate = '0.2 mA / $ cm^{2}$'
#CRate = '2.0 mA / $ cm^{2}$'
CRate = 'C/10'
#CRate = '2.5 mA / $\ cm^{2}$ resp. 1C'

#Name des Dictionaries
dictionary_name = 'dictionary_HIPOLE.txt'

#Einstellen Schriftart
#plt.rcParams.update({'font.family':'Times New Roman'})
#plt.rcParams.update({'font.size': 10})
#plt.rcParams.update({'mathtext.default': 'regular' })

###################################################################

if not isinstance(CycleNumber, (list, tuple)):      # macht aus CycleNumber eine Liste mit int
    CycleNumber = [CycleNumber]

header_row = 93                                     # Headerrow aus Biologic File, Default ohne Änderung des Messprogrammes während Betrieb: 93

# Geht durch alle Dateien und ermittelt maximale Anzahl an Zeilen (max_rows) --> damit danach ausreichend große Dictionaries erzeugt werden können
max_rows = 0
for filename in data_file_names:
    df_tmp = pd.read_table(filepath_or_buffer=data_path / filename,
        sep='\t',
        header=header_row,
        decimal=',',
        encoding='cp1252'
    )
    if df_tmp.shape[0] > max_rows:                  # Wenn längere Datei gefunden wird, wird max_rows neu gesetzt
        max_rows = df_tmp.shape[0]

# Dictionaries, um Daten für alle Zyklen zu speichern
x_values_charge_all = {cycle: pd.DataFrame(index=np.arange(max_rows)) for cycle in CycleNumber}
y_values_charge_all = {cycle: pd.DataFrame(index=np.arange(max_rows)) for cycle in CycleNumber}
x_values_discharge_all = {cycle: pd.DataFrame(index=np.arange(max_rows)) for cycle in CycleNumber}
y_values_discharge_all = {cycle: pd.DataFrame(index=np.arange(max_rows)) for cycle in CycleNumber}

# Dictionaries für Ableitungen
dq_dv_all = {cycle: {} for cycle in CycleNumber}
dv_dq_all = {cycle: {} for cycle in CycleNumber}
dq_dv_charge_all = {cycle: {} for cycle in CycleNumber}
dq_dv_discharge_all = {cycle: {} for cycle in CycleNumber}

# Bestimmung header line (wird gerade nicht weiter verwendet)
# Bestimmung Aktivmaterialgewicht --> wird in weight_list gespeichert
header_lines = []
header_line_list = []
weight_list = []
mode_line = None
mass_value = None


# Schreiben der Liste header_line_list mit
for counter_var in range(0,len(data_file_names)):
    with open(data_path / data_file_names[counter_var], "r", encoding="cp1252") as f:       # header steht immer nach : in zweiten Zeile des Files
            first_line = f.readline()
            nb_header_line_str = f.readline().strip()
            nb_header_lines = int(nb_header_line_str.split(":")[1].strip())
            header_line_list.append(nb_header_lines)
        # Read the header block
    with open(data_path / data_file_names[counter_var], "r", encoding="cp1252") as f:      # Aktive Masse steht in Zeile "mass of active material", nach : und ohne einheit mg
            for i in range(nb_header_lines):
                line = f.readline().strip()
                header_lines.append(line)
                # Find Mass of active material
                if "mass of active material" in line.lower():
                    # Extract number (assuming it's like "... : 5.0 mg")
                    parts = line.split(":")
                    # Den Zahlen-Teil isolieren und von mg befreien
                    number_str = parts[1].strip().split(' ')[0]
                    # Komma durch Punkt ersetzen, um die Zahl als float zu interpretieren
                    number_float = float(number_str.replace(',', '.'))
                    number_float = number_float/1000                                        # Umrechunung von mg in g
                    # In eine Liste packen
                    weight_list.append(number_float)

for idx, filename in enumerate(data_file_names):
    # Lese Datei mit Pandas ein
    df = pd.read_table(
        filepath_or_buffer=data_path / filename,
        sep='\t',
        header=header_row,
        decimal=',',
        encoding='cp1252'
    )

    # Spaltennamen vereinheitlichen
    df.rename(columns={'Ecell/V': 'Ecell', 'Capacity/mA.h': 'Capacity', 'half cycle': 'Half_cycle'}, inplace=True)

    # Daten vor erstem Ns-Wechsel entfernen, falls Spalte 'Ns' existiert
    # Sind Daten aus Rest-Schritt
    # geht nicht einfach über Zeile "half cycle" da dieser nicht wechselt zwischen Rest und 1. Entladezyklus
    if 'Ns' in df.columns:
        first_ns_value = df['Ns'].iloc[0]
        start_index = (df['Ns'] != first_ns_value).idxmax()
        df = df.loc[start_index:].reset_index(drop=True)

    # Prüfe ob 'Half_cycle' vorhanden ist
    if 'Half_cycle' not in df.columns:
        print(f"Warnung: 'Half_cycle' Spalte fehlt in Datei {filename}, Datei wird übersprungen.")
        continue

    # Gruppiere nach Half_cycle, Zyklen werden aufgeteilt
    grouped = df.groupby('Half_cycle')

    cycle_separated_df = pd.DataFrame() # leerer Dataframe

    for i, (half_cycle_value, group) in enumerate(grouped):
        cycle_num = (i // 2) + 1    # Zyklusnummer, half cycle beginnt: 0 für 1st DC, 2 für 1st CC, 3 für 2nd DC, 4 für 2nd CC
                                    # Ausnahme 0, sonst DC ungerade, CC gerade

        # Bestimme Spaltennamen basierend auf charge/discharge
        if i == 0:
            volt_col = f'disVolt{cycle_num}'
            cap_col = f'disCap{cycle_num}'
        else:
            if half_cycle_value % 2 != 0:  # ungerade Half_cycle: discharge
                volt_col = f'disVolt{cycle_num}'
                cap_col = f'disCap{cycle_num}'
            else:  # gerade Half_cycle: charge
                volt_col = f'chVolt{cycle_num}'
                cap_col = f'chCap{cycle_num}'

        cycle_separated_df[volt_col] = group['Ecell'].reset_index(drop=True)
        cycle_separated_df[cap_col] = group['Capacity'].reset_index(drop=True)

    # Normierung und Ableitung für die angegebenen Zyklen
    for cycle in CycleNumber:
        for filename in data_file_names:
            # Lade-Daten für Charge
            chCap_col = f'chCap{cycle}'
            chVolt_col = f'chVolt{cycle}'

            # Entlade-Daten für Discharge
            disCap_col = f'disCap{cycle}'
            disVolt_col = f'disVolt{cycle}'

            weight = weight_list[data_file_names.index(filename)]

            if chCap_col in cycle_separated_df.columns and chVolt_col in cycle_separated_df.columns:
                cap_norm = cycle_separated_df[chCap_col] / weight
                volt = cycle_separated_df[chVolt_col]

                # Gradient numerisch berechnen
                dq_dv = np.gradient(cap_norm) / np.gradient(volt)

                dq_dv_charge_all[cycle][filename] = (volt, dq_dv)

            if disCap_col in cycle_separated_df.columns and disVolt_col in cycle_separated_df.columns:
                cap_norm = cycle_separated_df[disCap_col] / weight
                volt = cycle_separated_df[disVolt_col]

                dq_dv = np.gradient(cap_norm) / np.gradient(volt)

                dq_dv_discharge_all[cycle][filename] = (volt, dq_dv)


            if chCap_col in cycle_separated_df.columns and chVolt_col in cycle_separated_df.columns:
                x_values_charge_all[cycle][filename] = cycle_separated_df[
                                                           chCap_col] / weight  # normierung Capa auf mAh/g
                y_values_charge_all[cycle][filename] = cycle_separated_df[chVolt_col]
            else:
                print(f"{filename}: Lade-Daten für Zyklus {cycle} nicht gefunden.")

            if disCap_col in cycle_separated_df.columns and disVolt_col in cycle_separated_df.columns:
                x_values_discharge_all[cycle][filename] = cycle_separated_df[
                                                              disCap_col] / weight  # normierung Capa auf mAh/g
                y_values_discharge_all[cycle][filename] = cycle_separated_df[disVolt_col]
            else:
                print(f"{filename}: Entlade-Daten für Zyklus {cycle} nicht gefunden.")


# Dictionary mit Infos über Batches -> für  Legenden-Beschriftung
file = open(dictionary_name, "r")
contents = file.read()
dic_legend_list = ast.literal_eval(contents)
file.close()

# Vergleicht Einträge aus batch_name_list nach einander mit allen keys aus dictionary und erzeugt Legenden Liste
legend_list = []
for ele in data_file_names:
    for key in dic_legend_list.keys():
        if key in ele:
            baustein1 = str(dic_legend_list[key][0])
            baustein2 = str(dic_legend_list[key][1])
            baustein3 = str(dic_legend_list[key][2])
            baustein4 = str(dic_legend_list[key][3]+'] mAh/$\\ cm^{2}$')
            #legend_list.append('{} (loading [{}$\pm${})'.format(baustein1, baustein3, baustein4))
            legend_list.append('{}'.format(baustein1))
if len(legend_list) != len(data_file_names):
    print('Not all batches were found in dictionary')
    sys.exit()


# Definition einer Farbliste, sonst haben Charge und Discharge der gleichen Zelle nicht gleiche Farbe
color_list = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive', 'tab:cyan',
             'black','indianred','lightgreen','orchid','gold','khaki','tab:pink','tab:gray','tab:olive', 'tab:cyan',
            'tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive', 'tab:cyan',
           'tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray','tab:olive', 'tab:cyan']
linestyle_list = ['--', '-', '--', '-']
alpha_list = [1, 0.2, 0.8, 0.6, 0.4, 0.2]

# Neuer Plot: dQ/dV vs Voltage
fig1, ax1 = plt.subplots()

for i, cycle in enumerate(CycleNumber):

    for col_idx, filename in enumerate(data_file_names):
        if filename in x_values_charge_all[cycle].columns:
            ax1.plot(
                y_values_charge_all[cycle][filename],
                dq_dv_charge_all[cycle][filename],
                color=color_list[col_idx],
                #linestyle=linestyle_list[col_idx],
                alpha=alpha_list[i],
                #label=f'{filename} Charge Cycle {cycle}',
                label=legend_list[col_idx] + '(Cycle {cycle})'.format(cycle=cycle)
            )
        if filename in x_values_discharge_all[cycle].columns:
            ax1.plot(
                y_values_discharge_all[cycle][filename],
                dq_dv_discharge_all[cycle][filename],
                color=color_list[col_idx],
                #linestyle=linestyle_list[col_idx],
                alpha=alpha_list[i],
                #label=f'{filename} Discharge Cycle {cycle}',
                #label=legend_list[col_idx]
            )
ax1.set_xlabel('Capacity [mAh $\\ g^{-1}$]')
ax1.set_ylabel('Voltage [V]')
ax1.legend(fontsize=10)
ax1.grid()
ax1.set_title(f'Cycles {CycleNumber} ({CRate})')
ax1.autoscale()

plt.savefig(os.path.join(str(data_path.parent), str(str(CycleNumber)+str(data_path.parts[-2]))))
plt.savefig(os.path.join(str(data_path.parent), str(str(CycleNumber)+str(data_path.parts[-2])+'.svg')))
plt.savefig(os.path.join(str(data_path.parent), str(str(CycleNumber)+str(data_path.parts[-2])+'.pdf')))
pickle.dump(fig, open((os.path.join(str(data_path.parent), data_path.parts[-2]) +'.pickle'), 'wb'))
