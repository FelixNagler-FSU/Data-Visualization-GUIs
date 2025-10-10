import yadg
import xarray
import os
import matplotlib
import json
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt


# Use 'Qt5Agg' backend for interactive plotting.
matplotlib.use('Qt5Agg')


# --- Konfiguration ---
# 1. Pfad zur EC-Lab .mpr-Datei
mpr_dateipfad = r"C:\Users\ro45vij\Desktop\AA_Data-Processing\AA_Plotting\MPR-Test\data\FN_E5_CC_16_C14.mpr"

# 2. Name für die Zieldatei im NetCDF-Format (.nc)
netcdf_dateipfad = r"C:\Users\ro45vij\Desktop\AA_Data-Processing\AA_Plotting\yadg-test\output_daten.nc"

# Dateityp, wie er von yadg erwartet wird
dateityp = 'eclab.mpr'

# --- Einlesen und Speichern ---
if not os.path.exists(mpr_dateipfad):
    print(f"Fehler: Quelldatei nicht gefunden unter '{mpr_dateipfad}'")
else:
    # 1. Daten mit yadg extrahieren
    # yadg gibt ein xarray.Dataset oder ein datatree.DataTree Objekt zurück.
    raw_data = yadg.extractors.extract(
        filetype=dateityp,
        path=mpr_dateipfad
    )
    print("\nDaten erfolgreich extrahiert (Typ: xarray.Dataset/DataTree).")

    original_metadata_string = raw_data.attrs['original_metadata']

    # 2. Den JSON-String in ein Python-Dictionary umwandeln (PARSING!)
    # *Nach* dieser Zeile ist 'original_metadata_dict' ein Dictionary
    original_metadata_dict = json.loads(original_metadata_string)
    try:
        mass_value = original_metadata_dict['settings']['active_material_mass']
        #print(f"Die aktive Materialmasse beträgt: {mass_value}")
    except KeyError as e:
        print(f"Fehler beim Zugriff auf den Schlüssel: {e}")

try:
    raw_data_ds = raw_data.to_dataset()
except Exception:
    try:
        raw_data_ds = raw_data["/"].to_dataset()
    except Exception:
        raise RuntimeError("Konnte raw_data nicht in xarray.Dataset umwandeln. raw_data überprüfen.")


q_charge_discharge = raw_data_ds['Q charge or discharge'].values
half_cycles = raw_data_ds['half cycle'].values

# 3) pandas DataFrame, gruppieren, letzte charge/discharge pro half
df = pd.DataFrame({'Q': np.asarray(q_charge_discharge, dtype=float), 'half_cycle_values': half_cycles})
# letzte positive (charge) pro half:
last_charge = df[df['Q'] > 0].groupby('half_cycle_values', sort=False).last()['Q']
# letzte negative (discharge) pro half, als positiver Betrag:
last_discharge = df[df['Q'] < 0].groupby('half_cycle_values', sort=False).last()['Q'].abs()
#Ns_changes = raw_data['Ns changes']
#half_cycle_var = raw_data['half cycle']


# Setze diese Option nach Deinem Bedarf:
first_cycle_discharge_only = True   # True = erste Halbzyklen sind Discharge-only

# Sammle die half-Keys in erster Auftretensreihenfolge
half_keys = list(dict.fromkeys(list(last_charge.index) + list(last_discharge.index)))

# Dictionaries für die Zuordnung cycle -> value
charge_by_cycle = {}
discharge_by_cycle = {}



for h in half_keys:
    # versuche, den half-key als Integer zu interpretieren
    try:
        h_int = int(h)
    except Exception:
        # überspringe nicht-integer keys
        continue

    # Berechne cycle und is_charge nach deinen Regeln
    if first_cycle_discharge_only:
        if h_int == 0:
            cycle = 1
            is_charge = False
        else:
            cycle = (h_int - 2) // 2 + 2
            is_charge = (h_int % 2 == 0)
    else:
        cycle = h_int // 2 + 1
        is_charge = (h_int % 2 == 0)

    # Hole den Wert aus den Series (falls vorhanden)
    if is_charge:
        val = last_charge.get(h, float('nan'))
        if not (val is None or (isinstance(val, float) and math.isnan(val))):
            # falls mehrere half in denselben cycle mappen, behalten wir die erste gefundene nicht-NaN
            if cycle not in charge_by_cycle:
                charge_by_cycle[cycle] = val
    else:
        val = last_discharge.get(h, float('nan'))
        if not (val is None or (isinstance(val, float) and math.isnan(val))):
            if cycle not in discharge_by_cycle:
                discharge_by_cycle[cycle] = val

# Erstelle sortiertes DataFrame mit cycles als Index
all_cycles = sorted(set(list(charge_by_cycle.keys()) + list(discharge_by_cycle.keys())))
df_cycles = pd.DataFrame(index=all_cycles, columns=['charge', 'discharge'], dtype=float)

for c in all_cycles:
    df_cycles.loc[c, 'charge'] = charge_by_cycle.get(c, float('nan'))
    df_cycles.loc[c, 'discharge'] = discharge_by_cycle.get(c, float('nan'))
df_cycles.index.name = 'cycle'

# Optional: Konvertiere Q (Coulomb) -> mAh/g, falls mass_value vorhanden (mass_value in mg)
try:
    mass_g = float(mass_value) / 1000.0   # mg -> g
    if mass_g <= 0:
        raise ValueError("mass_g <= 0")

    # Q in C -> mAh: divide by 3.6
    df_cycles['charge_mAh_g'] = (df_cycles['charge'].astype(float) / 3.6) / mass_g
    df_cycles['discharge_mAh_g'] = (df_cycles['discharge'].astype(float) / 3.6) / mass_g
    df_cycles['columbic_efficiency'] = (df_cycles['discharge'] / df_cycles['charge']) * 100.0
except Exception as e:
    # mass_value nicht verfügbar oder fehlerhaft: wir überspringen die Umrechnung
    df_cycles['charge_mAh_g'] = float('nan')
    df_cycles['discharge_mAh_g'] = float('nan')

# Kurze Ausgabe zur Kontrolle
print("Zuordnung abgeschlossen. Erste 10 Zyklen:")
print(df_cycles.head(10))


print('Hello II', q_charge_discharge)
print('Hello III', last_charge)
print('Hello IV', last_discharge)
