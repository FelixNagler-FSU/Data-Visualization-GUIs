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
import matplotlib.font_manager as font_manager
import pickle

# --- User Input Section ---
###################################################################
# Define the path to the directory containing the raw data files.
data_path = Path(r"C:\Users\ro45vij\Desktop\AA_Data-Processing\AA_Plotting\E5_Vergleich_Swagelok-CC\data")
# Get a list of all filenames in the data directory.
data_file_names = os.listdir(data_path)

# Define the script directory and the dictionary file for legend information.
script_dir = Path(r"C:\Users\ro45vij\PycharmProjects\Auswertung")
dictionary_name = script_dir / 'dictionary_HIPOLE.txt'

# Define the number of different batches to be imported and analyzed.
different_batches = 4

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
specific_charge_capacity_list = []
specific_discharge_capacity_list = []

# --- Data Loading and Preprocessing Loop ---
# This loop iterates through each data file to extract and process the relevant data.
for counter_var in range(0, len(data_file_names)):
    file_path = data_path / data_file_names[counter_var]
    filename = data_file_names[counter_var]

    # Dynamically read the actual header row by searching for a specific keyword.
    header_line_number = None
    try:
        with open(file_path, "r", encoding="cp1252") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                # Find the line containing the header and determine its index.
                if "mode	ox/red	error" in line.strip():
                    header_line_number = i - 3
                # Read header information to find the active material weight.
                if "mass of active material" in line.lower():
                    try:
                        parts = line.split(":")
                        number_str = parts[1].strip().split(' ')[0]
                        # Replace comma with period and convert to kilograms.
                        number_float = float(number_str.replace(',', '.')) / 1000
                        weight_dict[filename] = number_float
                        print(f"Active mass for {filename}: {number_float * 1000} mg")
                    except (IndexError, ValueError):
                        print(f"Warning: Could not read active mass in {filename}. Skipping file.")
                        break

            # If no weight was found in the header, skip the file.
            if filename not in weight_dict:
                print(f"Warning: No active mass found in header for {filename}. Skipping file.")
                continue
    except (IOError, IndexError, ValueError):
        print(f"Warning: Could not find header row or active mass in {filename}. Skipping file.")
        continue

    # If the header row was not found, skip the file.
    if header_line_number is None:
        print(f"Warning: Header row 'mode ox/red error' not found in {filename}. Skipping file.")
        continue

    # Read the main data from the file using the dynamically determined header row.
    data_df = pd.read_table(
        filepath_or_buffer=file_path,
        sep='\t',
        header=header_line_number,
        decimal=',',
        encoding='cp1252'
    )

    # Keep track of the maximum number of rows among all files.
    if data_df.shape[0] > max_rows:
        max_rows = data_df.shape[0]

    # Rename columns for easier access.
    data_df.rename(columns={'Q discharge/mA.h': 'DisCap', 'Q charge/mA.h': 'ChCap', 'half cycle': 'Half_cycle'},
                   inplace=True)

    # Find the indices corresponding to the end of each half-cycle.
    half_cycles = data_df['Half_cycle']
    half_cycles_diff = half_cycles.diff(periods=1)
    cycle_index = half_cycles_diff.loc[half_cycles_diff > 0.5].index - 1

    # Extract capacity values at the end of each half-cycle.
    filtered_discharge_capacity = data_df['DisCap'].loc[cycle_index]
    filtered_charge_capacity = data_df['ChCap'].loc[cycle_index]

    # Get the correct active mass for the current file from the dictionary.
    weight = weight_dict[data_file_names[counter_var]]

    # Convert capacity to specific capacity by dividing by the active mass.
    filtered_discharge_capacity = filtered_discharge_capacity.loc[filtered_discharge_capacity > 0.0] / weight
    filtered_charge_capacity = filtered_charge_capacity.loc[filtered_charge_capacity > 0.0] / weight

    # --- Apply new option: Set first charge capacity to NaN if the first cycle is discharge only ---
    if first_cycle_discharge_only.lower() == 'yes' and len(filtered_charge_capacity) > 0:
        filtered_charge_capacity.iloc[0] = np.nan

    # Append the processed data to the respective lists.
    specific_discharge_capacity_list.append(filtered_discharge_capacity.reset_index(drop=True))
    specific_charge_capacity_list.append(filtered_charge_capacity.reset_index(drop=True))

# --- Dataframe Creation and Cleanup ---
# Find the maximum length of all imported data to ensure uniform DataFrame size.
max_len = 0
for series in specific_discharge_capacity_list:
    if len(series) > max_len:
        max_len = len(series)

# Pad all data series with NaN values to a uniform length.
padded_discharge_list = []
padded_charge_list = []

for series in specific_discharge_capacity_list:
    padded_series = series.reindex(range(max_len), fill_value=np.nan)
    padded_discharge_list.append(padded_series)

for series in specific_charge_capacity_list:
    padded_series = series.reindex(range(max_len), fill_value=np.nan)
    padded_charge_list.append(padded_series)

# Create the final DataFrames from the padded lists.
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
# This loop calculates the mean and standard deviation for cells within the same batch.
# Pandas' mean() and std() functions automatically ignore NaN values.
for counter_var in range(0, different_batches):
    start_index = cell_numeration[counter_var]
    end_index = cell_numeration[counter_var + 1]

    # Calculate mean and stddev for discharge capacity.
    specific_discharge_capacity[f'{data_file_names[start_index]} mean discharge capacity'] = \
        specific_discharge_capacity.iloc[:, start_index:end_index].mean(axis=1)

    specific_discharge_capacity[f'{data_file_names[start_index]} stddev discharge capacity'] = \
        specific_discharge_capacity.iloc[:, start_index:end_index].std(axis=1)

    # Calculate mean and stddev for charge capacity.
    specific_charge_capacity[f'{data_file_names[start_index]} mean charge capacity'] = \
        specific_charge_capacity.iloc[:, start_index:end_index].mean(axis=1)

    specific_charge_capacity[f'{data_file_names[start_index]} stddev charge capacity'] = \
        specific_charge_capacity.iloc[:, start_index:end_index].std(axis=1)

    # Calculate mean and stddev for Coulombic Efficiency.
    coulombic_efficiency[f'{data_file_names[start_index]} mean'] = \
        coulombic_efficiency.iloc[:, start_index:end_index].mean(axis=1)

    coulombic_efficiency[f'{data_file_names[start_index]} stddev'] = \
        coulombic_efficiency.iloc[:, start_index:end_index].std(axis=1)

# --- Plotting Setup and Execution ---
# Create the list of legend labels based on the batch information.
columns_name_list = list(specific_discharge_capacity.columns.values)
batch_names_list = [x for x in columns_name_list if 'mean discharge capacity' in x]
legend_list = []
for ele in batch_names_list:
    for key in legend_info_dict.keys():
        if key in ele:
            element1 = str(legend_info_dict[key][0])
            legend_list.append(f'{element1} ')
if len(legend_list) != different_batches:
    print('Error: Not all batches were found in dictionary')
    import sys

    sys.exit()

# Add a 'Cycle' column for plotting.
specific_discharge_capacity['Cycle'] = np.arange(1, specific_discharge_capacity.shape[0] + 1)
specific_charge_capacity['Cycle'] = np.arange(1, specific_charge_capacity.shape[0] + 1)
coulombic_efficiency['Cycle'] = np.arange(1, coulombic_efficiency.shape[0] + 1)

# Adjust plot boundaries to the actual number of cycles.
max_cycle_dis = specific_discharge_capacity.shape[0]
max_cycle_ce = coulombic_efficiency.shape[0]

# Define the y-axis label based on the selected plot mode.
if capacity_plot_mode == 'discharge':
    ylabel_text = 'Specific Discharge Capacity [mAh $g^{-1}$]'
elif capacity_plot_mode == 'charge':
    ylabel_text = 'Specific Charge Capacity [mAh $g^{-1}$]'
else:
    ylabel_text = 'Specific Capacity [mAh $g^{-1}$]'

# Define the lists of colors, markers, and line styles for plotting.
color_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan',
              'black', 'indianred', 'lightgreen', 'orchid', 'gold', 'khaki', 'tab:pink', 'tab:gray', 'tab:olive',
              'tab:cyan',
              'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan',
              'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
              'tab:olive', 'tab:cyan']

marker_list = ["o", "v", "^", "<", ">", "s", "p", "2", "3", "4", "8", "s", "p", "P", "*", "h", "H", "+", "x", "X", "D",
               "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
linestyle_list = ['--', '-', '--', '-']
fillstyle_list = ['none', 'full', 'none', 'full']

# Plot the data based on the user's 'ce_graph' input.
if ce_graph.lower() == 'yes':
    # Create a figure with two subplots: one for CE and one for capacity.
    fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(7, 5))
    for counter_var in range(0, different_batches):
        if capacity_plot_mode in ['discharge', 'both']:
            # Plot the discharge capacity with error bars.
            label_dis = f'{legend_list[counter_var]} (Discharge)' if capacity_plot_mode == 'both' else f'{legend_list[counter_var]}'
            axs[1].errorbar(specific_discharge_capacity['Cycle'],
                            specific_discharge_capacity[
                                '{} mean discharge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                            specific_discharge_capacity[
                                '{} stddev discharge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                            label=label_dis
                            , capsize=2.5
                            , marker=marker_list[start_marker + counter_var]
                            , color=color_list[start_color + counter_var]
                            , markersize=6
                            , errorevery=10
                            )

        if capacity_plot_mode in ['charge', 'both']:
            # Plot the charge capacity with error bars.
            label_ch = f'{legend_list[counter_var]} (Charge)' if capacity_plot_mode == 'both' else f'{legend_list[counter_var]}'
            axs[1].errorbar(specific_charge_capacity['Cycle'],
                            specific_charge_capacity[
                                '{} mean charge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                            specific_charge_capacity[
                                '{} stddev charge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                            label=label_ch
                            , capsize=2.5
                            , marker=marker_list[start_marker + counter_var]
                            , color=color_list[start_color + counter_var]
                            , linestyle='--'
                            , fillstyle='none'
                            , markersize=6
                            , errorevery=10
                            )

        # Plot the Coulombic Efficiency with error bars.
        axs[0].errorbar(coulombic_efficiency['Cycle'],
                        coulombic_efficiency['{} mean'.format(data_file_names[cell_numeration[counter_var]])],
                        coulombic_efficiency['{} stddev'.format(data_file_names[cell_numeration[counter_var]])],
                        capsize=2.5
                        , errorevery=10
                        , marker=marker_list[start_marker + counter_var]
                        , markersize=6
                        , color=color_list[start_color + counter_var]
                        )

    # Set labels, limits, and grid for the capacity subplot.
    axs[1].set_xlabel('Cycle')
    axs[1].set_ylabel(ylabel_text)
    axs[1].legend(fontsize=12, loc=0)
    axs[1].set_xlim(0, max_cycle_dis + 1)
    axs[1].set_ylim(0, 210)
    axs[1].grid()
    axs[1].autoscale()

    # Set labels, limits, and grid for the CE subplot.
    axs[0].set_xlim(0, max_cycle_ce + 1)
    axs[0].set_xticklabels([])
    axs[0].set_ylabel('CE [%]')
    axs[0].set_ylim(98, 102)
    axs[0].grid()
    axs[0].autoscale()
    plt.tight_layout()

else:
    # Create a single subplot for capacity only.
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for counter_var in range(0, different_batches):
        if capacity_plot_mode in ['discharge', 'both']:
            # Plot discharge capacity.
            label_dis = f'{legend_list[counter_var]} (Discharge)' if capacity_plot_mode == 'both' else f'{legend_list[counter_var]}'
            ax.errorbar(specific_discharge_capacity['Cycle'],
                        specific_discharge_capacity[
                            '{} mean discharge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                        specific_discharge_capacity[
                            '{} stddev discharge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                        label=label_dis
                        , capsize=2.5
                        , marker=marker_list[start_marker + counter_var]
                        , color=color_list[start_color + counter_var]
                        )

        if capacity_plot_mode in ['charge', 'both']:
            # Plot charge capacity.
            label_ch = f'{legend_list[counter_var]} (Charge)' if capacity_plot_mode == 'both' else f'{legend_list[counter_var]}'
            ax.errorbar(specific_charge_capacity['Cycle'],
                        specific_charge_capacity[
                            '{} mean charge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                        specific_charge_capacity[
                            '{} stddev charge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                        label=label_ch
                        , capsize=2.5
                        , marker=marker_list[start_marker + counter_var]
                        , color=color_list[start_color + counter_var]
                        , linestyle='--'
                        , fillstyle='none'
                        )

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