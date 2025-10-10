import pandas as pd
import numpy as np
import scipy as sp
import matplotlib

# Use 'Qt5Agg' backend for interactive plotting.
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
from pathlib import Path
import os
import ast
import json
import matplotlib.font_manager as font_manager
import pickle

# --- User Input Section ---
###################################################################
# Define the path to the directory containing the raw data files.
data_path = Path(r"C:\Users\ro45vij\Desktop\AA_Data-Processing\AA_Plotting\MPR-Test\data")
# Get a list of all filenames in the data directory.
data_file_names = os.listdir(data_path)
# Define the path to the directory (or single file) containing the raw data.
#data_path = Path(r"C:\Users\ro45vij\Desktop\AA_Data-Processing\AA_Plotting\E5\data_neu")

# Allow data_path to be either a directory or a single file path.
if data_path.is_dir():
    data_file_names = os.listdir(data_path)
elif data_path.is_file():
    # user pointed data_path to a single file -> process that one file
    data_file_names = [data_path.name]
    data_path = data_path.parent
else:
    raise NotADirectoryError(f"Data path is not a directory or file: {data_path}")

# Define the script directory and the dictionary file for legend information.
script_dir = Path(r"C:\Users\ro45vij\PycharmProjects\Auswertung")
dictionary_name = script_dir / 'dictionary_HIPOLE.txt'

# Define the number of different batches to be imported and analyzed.
different_batches = 2

# Define the number of cells measured for each batch.
# This list corresponds to the number of files per batch, which is used to group the data.
number_of_cells = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2]

# Plot graph with or without Coulombic Efficiency (CE). 'Yes' = 1, 'No' = 0.
ce_graph = 'Yes'

# Define which capacity curves should be plotted: 'discharge', 'charge', or 'both'.
capacity_plot_mode = 'both'

# --- NEW OPTION: Flag to indicate if the first cycle is discharge only ---
first_cycle_discharge_only = 'Yes'

# Define the starting index for plot colors and markers.
start_color = 0
start_marker = 0

data_type = 'eclab.mpr'

# Set global font style parameters for the plots.
# plt.rcParams.update({'font.family':'Times New Roman'})
plt.rcParams.update({'font.size': 13})
# plt.rcParams.update({'mathtext.default': 'regular' })

# --- End of User Input Section ---
#############################################################

# Create a dictionary to store the active material mass for each cell's filename.
weight_dict = {}

# Read batch information for legend labeling from a dictionary file.
# The 'ast' module is used to safely parse the dictionary from a text file.
try:
    with open(dictionary_name, "r") as file:
        contents = file.read()
        legend_info_dict = ast.literal_eval(contents)
except FileNotFoundError:
    print(f"Error: Dictionary file not found at {dictionary_name}")
    exit()

# Calculate the indices where a new batch starts in the list of files.
# This helps in grouping the files for mean/stddev calculation.
cell_numeration = [sum(number_of_cells[0:counter_var]) for counter_var in range(0, len(number_of_cells))]
cell_numeration.append(sum(number_of_cells))

# Initialize lists to hold the specific capacities for each file before creating the final DataFrames.
max_rows = 0
# Initialize lists to hold the specific capacities for each file before creating the final DataFrames.
max_rows = 0
specific_charge_capacity_list = []
specific_discharge_capacity_list = []

# (DataFrames will be constructed after parsing the files with yadg)

# --- Data Loading and Preprocessing Loop (use yadg to read .mpr files) ---
for counter_var in range(0, len(data_file_names)):
    file_path = data_path / data_file_names[counter_var]
    filename = data_file_names[counter_var]

    # Only process .mpr files
    if Path(filename).suffix.lower() != '.mpr':
        print(f"Skipping non-mpr file: {filename}")
        continue

    try:
        # try yadg parser
        import yadg
        ds_raw = yadg.extractors.extract(filetype='eclab.mpr', path=str(file_path))
    except Exception as e:
        print(f"Warning: failed to parse {filename} with yadg: {e}. Skipping.")
        continue

    # read active material mass from original_metadata (JSON-like)
    mass_g = None
    try:
        orig = getattr(ds_raw, 'attrs', {}).get('original_metadata', None)
        if orig is None and 'original_metadata' in ds_raw:
            orig = ds_raw['original_metadata']
        if isinstance(orig, (bytes, str)):
            try:
                orig = json.loads(orig)
            except Exception:
                try:
                    orig = ast.literal_eval(orig)
                except Exception:
                    orig = None
        # nested lookup
        def _find_key(d, key):
            if isinstance(d, dict):
                if key in d:
                    return d[key]
                for v in d.values():
                    res = _find_key(v, key)
                    if res is not None:
                        return res
            return None

        found = None
        if isinstance(orig, dict) and 'settings' in orig and 'active_material_mass' in orig['settings']:
            found = orig['settings']['active_material_mass']
        if found is None:
            found = _find_key(orig, 'active_material_mass')
        if found is not None:
            mg = float(found)
            mass_g = mg / 1000.0
            weight_dict[filename] = mass_g
            print(f"Active mass for {filename}: {mg} mg")
    except Exception as e:
        print(f"Warning: error reading metadata for {filename}: {e}")

    if mass_g is None:
        print(f"Warning: No active material mass found for {filename}. Skipping file.")
        continue

    # Convert datatree -> xarray Dataset if needed
    try:
        if hasattr(ds_raw, 'to_dataset'):
            ds = ds_raw.to_dataset()
        elif hasattr(ds_raw, 'to_xarray'):
            ds = ds_raw.to_xarray()
        else:
            # try root group
            if hasattr(ds_raw, 'get') and ds_raw.get('/') is not None:
                ds = ds_raw.get('/').to_dataset()
            else:
                ds = ds_raw
    except Exception:
        ds = ds_raw

    # find Q and half-cycle variables
    q_arr = None
    half_arr = None
    if 'Q charge or discharge' in ds:
        q_arr = np.asarray(ds['Q charge or discharge'].values, dtype=float)
    elif 'Q charge/discharge' in ds:
        q_arr = np.asarray(ds['Q charge/discharge'].values, dtype=float)
    if 'half cycle' in ds:
        half_arr = np.asarray(ds['half cycle'].values)
    elif 'Half_cycle' in ds:
        half_arr = np.asarray(ds['Half_cycle'].values)

    if q_arr is None or half_arr is None:
        print(f"Warning: Q or half-cycle variables not found in {filename}. Skipping.")
        continue

    # Build a dataframe and take last sample per half-cycle
    tmp_df = pd.DataFrame({'Q': q_arr, 'half': half_arr})
    # last positive (charge) per half
    last_charge = tmp_df[tmp_df['Q'] > 0].groupby('half', sort=False).last()['Q']
    # last negative (discharge) per half as positive value
    last_discharge = tmp_df[tmp_df['Q'] < 0].groupby('half', sort=False).last()['Q'].abs()
    # Map half-cycle keys to cycles (keep first-occurrence order) and collect one charge/discharge value per cycle
    first_cycle_discharge_only_bool = True if str(first_cycle_discharge_only).lower() in ('yes', 'true', '1') else False

    # preserve order: combine indices from charge and discharge
    half_keys = list(dict.fromkeys(list(last_charge.index) + list(last_discharge.index)))

    charge_by_cycle = {}
    discharge_by_cycle = {}

    for h in half_keys:
        try:
            h_int = int(h)
        except Exception:
            # skip non-integer keys
            continue

        if first_cycle_discharge_only_bool:
            if h_int == 0:
                cycle = 1
                is_charge = False
            else:
                cycle = (h_int - 2) // 2 + 2
                is_charge = (h_int % 2 == 0)
        else:
            cycle = h_int // 2 + 1
            is_charge = (h_int % 2 == 0)

        if is_charge:
            val = last_charge.get(h, float('nan'))
            if not (val is None or (isinstance(val, float) and np.isnan(val))):
                if cycle not in charge_by_cycle:
                    charge_by_cycle[cycle] = val
        else:
            val = last_discharge.get(h, float('nan'))
            if not (val is None or (isinstance(val, float) and np.isnan(val))):
                if cycle not in discharge_by_cycle:
                    discharge_by_cycle[cycle] = val

    # assemble per-cycle DataFrame
    all_cycles = sorted(set(list(charge_by_cycle.keys()) + list(discharge_by_cycle.keys())))
    df_cycles = pd.DataFrame(index=all_cycles, columns=['charge', 'discharge'], dtype=float)
    for c in all_cycles:
        df_cycles.loc[c, 'charge'] = charge_by_cycle.get(c, float('nan'))
        df_cycles.loc[c, 'discharge'] = discharge_by_cycle.get(c, float('nan'))
    df_cycles.index.name = 'cycle'

    # Convert Q (Coulomb) -> mAh and normalize by active mass (mass_g is in g)
    try:
        df_cycles['charge_mAh_g'] = (df_cycles['charge'].astype(float) / 3.6) / float(mass_g)
    except Exception:
        df_cycles['charge_mAh_g'] = float('nan')
    try:
        df_cycles['discharge_mAh_g'] = (df_cycles['discharge'].astype(float) / 3.6) / float(mass_g)
    except Exception:
        df_cycles['discharge_mAh_g'] = float('nan')

    # Append per-file Series (indexed 0..N-1 after reset)
    specific_charge_capacity_list.append(df_cycles['charge_mAh_g'].reset_index(drop=True))
    specific_discharge_capacity_list.append(df_cycles['discharge_mAh_g'].reset_index(drop=True))

# --- Build DataFrames from parsed series ---
max_len = 0
for series in specific_discharge_capacity_list:
    if len(series) > max_len:
        max_len = len(series)

padded_discharge_list = []
padded_charge_list = []

for series in specific_discharge_capacity_list:
    padded_series = series.reindex(range(max_len), fill_value=np.nan)
    padded_discharge_list.append(padded_series)

for series in specific_charge_capacity_list:
    padded_series = series.reindex(range(max_len), fill_value=np.nan)
    padded_charge_list.append(padded_series)

specific_discharge_capacity = pd.DataFrame()
specific_charge_capacity = pd.DataFrame()

for i, filtered_dis_cap in enumerate(padded_discharge_list):
    specific_discharge_capacity[f'{data_file_names[i]} DisCap'] = filtered_dis_cap

for i, filtered_ch_cap in enumerate(padded_charge_list):
    specific_charge_capacity[f'{data_file_names[i]} ChCap'] = filtered_ch_cap

# Replace any zero values with NaN to avoid division by zero errors when calculating CE.
specific_charge_capacity_cleaned = specific_charge_capacity.replace(0, np.nan)
# Create copies of the DataFrames with harmonized column names for CE calculation.
discharge_for_ce = specific_discharge_capacity.copy()
charge_for_ce = specific_charge_capacity_cleaned.copy()

discharge_for_ce.columns = [col.replace(" DisCap", "") for col in discharge_for_ce.columns]
charge_for_ce.columns = [col.replace(" ChCap", "") for col in charge_for_ce.columns]

# Calculate the Coulombic Efficiency (CE) as (Discharge Capacity / Charge Capacity) * 100.
coulombic_efficiency = (discharge_for_ce / charge_for_ce) * 100

# --- Calculate Mean and Standard Deviation for each Batch ---
for counter_var in range(0, different_batches):
    start_index = cell_numeration[counter_var]
    end_index = cell_numeration[counter_var + 1]

    specific_discharge_capacity[f'{data_file_names[start_index]} mean discharge capacity'] = \
        specific_discharge_capacity.iloc[:, start_index:end_index].mean(axis=1)

    specific_discharge_capacity[f'{data_file_names[start_index]} stddev discharge capacity'] = \
        specific_discharge_capacity.iloc[:, start_index:end_index].std(axis=1)

    specific_charge_capacity[f'{data_file_names[start_index]} mean charge capacity'] = \
        specific_charge_capacity.iloc[:, start_index:end_index].mean(axis=1)

    specific_charge_capacity[f'{data_file_names[start_index]} stddev charge capacity'] = \
        specific_charge_capacity.iloc[:, start_index:end_index].std(axis=1)

    coulombic_efficiency[f'{data_file_names[start_index]} mean'] = \
        coulombic_efficiency.iloc[:, start_index:end_index].mean(axis=1)

    coulombic_efficiency[f'{data_file_names[start_index]} stddev'] = \
        coulombic_efficiency.iloc[:, start_index:end_index].std(axis=1)
# Build legend list and plotting defaults
columns_name_list = list(specific_discharge_capacity.columns.values)
batch_names_list = [x for x in columns_name_list if 'mean discharge capacity' in x]
legend_list = []
for ele in batch_names_list:
    for key in legend_info_dict.keys():
        if key in ele:
            element1 = str(legend_info_dict[key][0])
            legend_list.append(f'{element1} ')
if len(legend_list) != different_batches:
    print('Warning: Not all batches were found in dictionary')

# plotting defaults
color_list = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']
marker_list = ['o', 's', '^', 'D', 'v', '<', '>', 'x', '.', '+', '*']
ylabel_text = 'Specific capacity (mAh/g)'

# Add a Cycle column to each DataFrame so plotting routines can index by 'Cycle'
specific_discharge_capacity['Cycle'] = np.arange(1, specific_discharge_capacity.shape[0] + 1)
specific_charge_capacity['Cycle'] = np.arange(1, specific_charge_capacity.shape[0] + 1)
coulombic_efficiency['Cycle'] = np.arange(1, coulombic_efficiency.shape[0] + 1)

# max cycle counters used for axis limits
max_cycle_dis = specific_discharge_capacity.shape[0]
max_cycle_ce = coulombic_efficiency.shape[0]

if str(ce_graph).lower() == 'yes':
    # Create CE (top) + Capacity (bottom) subplots
    fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(8, 6))

    # Mean and stddev plots for Capacity (bottom)
    for counter_var in range(0, different_batches):
        if capacity_plot_mode in ['discharge', 'both']:
            label_dis = f'{legend_list[counter_var]} (Discharge)' if capacity_plot_mode == 'both' else f'{legend_list[counter_var]}'
            axs[1].errorbar(specific_discharge_capacity['Cycle'],
                            specific_discharge_capacity[ '{} mean discharge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                            specific_discharge_capacity[ '{} stddev discharge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                            label=label_dis, capsize=2.5,
                            marker=marker_list[(start_marker + counter_var) % len(marker_list)],
                            color=color_list[(start_color + counter_var) % len(color_list)],
                            markersize=6)

        if capacity_plot_mode in ['charge', 'both']:
            label_ch = f'{legend_list[counter_var]} (Charge)' if capacity_plot_mode == 'both' else f'{legend_list[counter_var]}'
            axs[1].errorbar(specific_charge_capacity['Cycle'],
                            specific_charge_capacity[ '{} mean charge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                            specific_charge_capacity[ '{} stddev charge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                            label=label_ch, capsize=2.5,
                            marker=marker_list[(start_marker + counter_var) % len(marker_list)],
                            color=color_list[(start_color + counter_var) % len(color_list)],
                            linestyle='--', fillstyle='none', markersize=6)

    # Plot CE (top)
    for counter_var in range(0, different_batches):
        try:
            mean_col = '{} mean'.format(data_file_names[cell_numeration[counter_var]])
            std_col = '{} stddev'.format(data_file_names[cell_numeration[counter_var]])
            axs[0].errorbar(coulombic_efficiency['Cycle'], coulombic_efficiency[mean_col], coulombic_efficiency[std_col],
                            capsize=2.5,
                            marker=marker_list[(start_marker + counter_var) % len(marker_list)],
                            markersize=6,
                            color=color_list[(start_color + counter_var) % len(color_list)])
        except Exception:
            # missing columns or data -> skip
            continue

    # Labels and layout
    axs[1].set_title('Capacity')
    axs[1].set_xlabel('Cycle')
    axs[1].set_ylabel(ylabel_text)
    axs[1].legend(fontsize=11, loc=0)
    axs[1].grid()
    if max_cycle_dis is not None:
        axs[1].set_xlim(0, max_cycle_dis + 1)

    axs[0].set_ylabel('CE [%]')
    axs[0].set_ylim(0, 200)
    axs[0].grid()
    axs[0].set_xticklabels([])
    plt.tight_layout()

else:
    # Create a single subplot for capacity only.
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for counter_var in range(0, different_batches):
        if capacity_plot_mode in ['discharge', 'both']:
            # Plot discharge capacity.
            label_dis = f'{legend_list[counter_var]} (Discharge)' if capacity_plot_mode == 'both' else f'{legend_list[counter_var]}'
            ax.errorbar(specific_discharge_capacity['Cycle'],
                        specific_discharge_capacity['{} mean discharge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                        specific_discharge_capacity['{} stddev discharge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                        label=label_dis, capsize=2.5,
                        marker=marker_list[(start_marker + counter_var) % len(marker_list)],
                        color=color_list[(start_color + counter_var) % len(color_list)])

        if capacity_plot_mode in ['charge', 'both']:
            # Plot charge capacity.
            label_ch = f'{legend_list[counter_var]} (Charge)' if capacity_plot_mode == 'both' else f'{legend_list[counter_var]}'
            ax.errorbar(specific_charge_capacity['Cycle'],
                        specific_charge_capacity['{} mean charge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                        specific_charge_capacity['{} stddev charge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                        label=label_ch, capsize=2.5,
                        marker=marker_list[(start_marker + counter_var) % len(marker_list)],
                        color=color_list[(start_color + counter_var) % len(color_list)],
                        linestyle='--', fillstyle='none')

    # Set labels, limits, and grid for the single capacity plot.
    ax.set_xlabel('Cycle')
    ax.set_ylabel(ylabel_text)
    ax.legend(fontsize=11, loc=0)
    ax.grid()
    ax.set_xlim(0, specific_discharge_capacity.shape[0] + 1)
    ax.autoscale()
    plt.tight_layout()

# --- Save Figures and Data ---
# Save the generated figure in various formats (PNG, SVG, and a pickle file).
plt.savefig(os.path.join(str(data_path.parent), data_path.parts[-2]))
plt.savefig(os.path.join(str(data_path.parent), str(str(data_path.name) + '_' + str(data_path.parts[-2]) + '.svg')))
plt.savefig(os.path.join(str(data_path.parent), str(str(data_path.name) + '_' + str(data_path.parts[-2]))))
pickle.dump(fig, open((os.path.join(str(data_path.parent), data_path.parts[-2]) + 'CE' + '.pickle'), 'wb'))

# Define the filename for the text file to save the processed data.
save_file_name = os.path.join(str(data_path.parent), str(str(data_path.name) + '_' + str(data_path.parts[-2]) + '.txt'))

# Create a new DataFrame to save the final data.
df_to_save = pd.DataFrame({'Cycle': np.arange(1, max_len + 1)})

# Add the mean and stddev values for discharge, charge, and CE to the DataFrame.
for counter_var in range(0, different_batches):
    batch_name_part = data_file_names[cell_numeration[counter_var]]
    df_to_save[f'{batch_name_part} mean discharge capacity'] = specific_discharge_capacity[
        f'{batch_name_part} mean discharge capacity']
    df_to_save[f'{batch_name_part} stddev discharge capacity'] = specific_discharge_capacity[
        f'{batch_name_part} stddev discharge capacity']
    df_to_save[f'{batch_name_part} mean charge capacity'] = specific_charge_capacity[
        f'{batch_name_part} mean charge capacity']
    df_to_save[f'{batch_name_part} stddev charge capacity'] = specific_charge_capacity[
        f'{batch_name_part} stddev charge capacity']
    df_to_save[f'{batch_name_part} mean CE'] = coulombic_efficiency[f'{batch_name_part} mean']
    df_to_save[f'{batch_name_part} stddev CE'] = coulombic_efficiency[f'{batch_name_part} stddev']

# Save the DataFrame to a text file. The na_rep='' argument ensures that NaN values are saved as empty fields.
df_to_save.to_csv(save_file_name, sep=',', index=False, na_rep='')
print(f"Data has been saved to {save_file_name}.")

# Show the plots.
plt.show()