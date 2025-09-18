import pandas as pd
import numpy as np
import scipy as sp
import matplotlib
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import os
import ast
import matplotlib.font_manager as font_manager
import pickle

# Use TkAgg backend to integrate with the tkinter GUI
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

print('Versionsänderung 17.09.2025, 16:13 Uhr')
print('Versionsänderung 17.09.2025, 16:26 Uhr')


# --- Refactored plotting logic into a function ---
def run_plotting(data_path_str, dictionary_name_str, different_batches_str, number_of_cells_str, ce_graph_str,
                 capacity_plot_mode_str, first_cycle_discharge_only_str, plot_individual_cells_str, color_list_str,
                 marker_list_str,
                 capacity_plot_title_str, capacity_ylabel_text_str, capacity_xmin_str, capacity_xmax_str,
                 capacity_ymin_str, capacity_ymax_str,
                 ce_plot_title_str, ce_ylabel_text_str, ce_xmin_str, ce_xmax_str, ce_ymin_str, ce_ymax_str,
                 individual_cell_legend_suffix_str):
    """
    This function contains the core plotting logic, refactored to accept user inputs.
    It processes the data and generates the plots based on the provided parameters.
    """

    # Convert string inputs to the required data types
    try:
        data_path = Path(data_path_str)
        dictionary_name = Path(dictionary_name_str)
        different_batches = int(different_batches_str)
        number_of_cells = [int(x.strip()) for x in number_of_cells_str.split(',')]
        ce_graph = ce_graph_str
        capacity_plot_mode = capacity_plot_mode_str
        first_cycle_discharge_only = first_cycle_discharge_only_str
        plot_individual_cells = plot_individual_cells_str
        color_list = [c.strip() for c in color_list_str.split(',')]
        marker_list = [m.strip() for m in marker_list_str.split(',')]
        capacity_plot_title = capacity_plot_title_str
        capacity_ylabel_text = capacity_ylabel_text_str
        ce_plot_title = ce_plot_title_str
        ce_ylabel_text = ce_ylabel_text_str
        individual_cell_legend_suffix = individual_cell_legend_suffix_str

        # Parse and handle optional numerical inputs
        capacity_xmin = float(capacity_xmin_str) if capacity_xmin_str else None
        capacity_xmax = float(capacity_xmax_str) if capacity_xmax_str else None
        capacity_ymin = float(capacity_ymin_str) if capacity_ymin_str else None
        capacity_ymax = float(capacity_ymax_str) if capacity_ymax_str else None
        ce_xmin = float(ce_xmin_str) if ce_xmin_str else None
        ce_xmax = float(ce_xmax_str) if ce_xmax_str else None
        ce_ymin = float(ce_ymin_str) if ce_ymin_str else None
        ce_ymax = float(ce_ymax_str) if ce_ymax_str else None

    except (ValueError, IndexError) as e:
        messagebox.showerror("Input Error", f"Please check your inputs. Error: {e}")
        return

    # Check if the directories and files exist
    if not data_path.is_dir():
        messagebox.showerror("Error", f"Data directory not found: {data_path}")
        return
    if not dictionary_name.is_file():
        messagebox.showerror("Error", f"Dictionary file not found: {dictionary_name}")
        return

    data_file_names = os.listdir(data_path)

    plt.rcParams.update({'font.size': 13})

    # --- Data Processing Section ---
    weight_dict = {}
    try:
        with open(dictionary_name, "r") as file:
            contents = file.read()
            legend_info_dict = ast.literal_eval(contents)
    except FileNotFoundError:
        messagebox.showerror("Error", f"Dictionary file not found: {dictionary_name}")
        return
    except (SyntaxError, ValueError) as e:
        messagebox.showerror("Error", f"Invalid format in dictionary file. Error: {e}")
        return

    if len(number_of_cells) != len(data_file_names):
        messagebox.showwarning("Warning",
                               "Number of cells per batch does not match the number of files. Processing will continue, but this might lead to errors.")

    cell_numeration = [sum(number_of_cells[0:counter_var]) for counter_var in range(0, len(number_of_cells))]
    cell_numeration.append(sum(number_of_cells))

    max_rows = 0
    specific_charge_capacity_list = []
    specific_discharge_capacity_list = []

    for counter_var in range(0, len(data_file_names)):
        file_path = data_path / data_file_names[counter_var]
        filename = data_file_names[counter_var]

        header_line_number = None
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
                            weight_dict[filename] = number_float
                            print(f"Active mass for {filename}: {number_float * 1000} mg")
                        except (IndexError, ValueError):
                            print(f"Warning: Could not read active mass in {filename}. Skipping file.")
                            break
                if filename not in weight_dict:
                    print(f"Warning: No active mass found in header for {filename}. Skipping file.")
                    continue
        except (IOError, IndexError, ValueError):
            print(f"Warning: Could not find header row or active mass in {filename}. Skipping file.")
            continue

        if header_line_number is None:
            print(f"Warning: Header row 'mode ox/red error' not found in {filename}. Skipping file.")
            continue

        try:
            data_df = pd.read_table(
                filepath_or_buffer=file_path,
                sep='\t',
                header=header_line_number,
                decimal=',',
                encoding='cp1252'
            )
        except Exception as e:
            print(f"Error reading file {filename}: {e}. Skipping.")
            continue

        if data_df.shape[0] > max_rows:
            max_rows = data_df.shape[0]

        data_df.rename(columns={'Q discharge/mA.h': 'DisCap', 'Q charge/mA.h': 'ChCap', 'half cycle': 'Half_cycle'},
                       inplace=True)

        half_cycles = data_df['Half_cycle']
        half_cycles_diff = half_cycles.diff(periods=1)
        cycle_index = half_cycles_diff.loc[half_cycles_diff > 0.5].index - 1

        filtered_discharge_capacity = data_df['DisCap'].loc[cycle_index]
        filtered_charge_capacity = data_df['ChCap'].loc[cycle_index]

        weight = weight_dict[data_file_names[counter_var]]

        filtered_discharge_capacity = filtered_discharge_capacity.loc[filtered_discharge_capacity > 0.0] / weight
        filtered_charge_capacity = filtered_charge_capacity.loc[filtered_charge_capacity > 0.0] / weight

        if first_cycle_discharge_only.lower() == 'yes' and len(filtered_charge_capacity) > 0:
            filtered_charge_capacity.iloc[0] = np.nan

        specific_discharge_capacity_list.append(filtered_discharge_capacity.reset_index(drop=True))
        specific_charge_capacity_list.append(filtered_charge_capacity.reset_index(drop=True))

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

    specific_charge_capacity_cleaned = specific_charge_capacity.replace(0, np.nan)
    discharge_for_ce = specific_discharge_capacity.copy()
    charge_for_ce = specific_charge_capacity_cleaned.copy()

    discharge_for_ce.columns = [col.replace(" DisCap", "") for col in discharge_for_ce.columns]
    charge_for_ce.columns = [col.replace(" ChCap", "") for col in charge_for_ce.columns]

    coulombic_efficiency = (discharge_for_ce / charge_for_ce) * 100

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

    specific_discharge_capacity['Cycle'] = np.arange(1, specific_discharge_capacity.shape[0] + 1)
    specific_charge_capacity['Cycle'] = np.arange(1, specific_charge_capacity.shape[0] + 1)
    coulombic_efficiency['Cycle'] = np.arange(1, coulombic_efficiency.shape[0] + 1)

    max_cycle_dis = specific_discharge_capacity.shape[0]
    max_cycle_ce = coulombic_efficiency.shape[0]

    # --- Plotting execution ---
    if ce_graph.lower() == 'yes':
        fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(7, 5))

        # Individual cell plots for Capacity
        if plot_individual_cells.lower() == 'yes':
            for counter_var in range(0, len(data_file_names)):
                label_prefix = f'Cell {counter_var + 1}'
                if individual_cell_legend_suffix:
                    label_prefix += f' {individual_cell_legend_suffix}'

                if capacity_plot_mode in ['discharge', 'both']:
                    axs[1].plot(specific_discharge_capacity['Cycle'],
                                specific_discharge_capacity[f'{data_file_names[counter_var]} DisCap'],
                                label=f'{label_prefix} (Discharge)' if capacity_plot_mode == 'both' else label_prefix,
                                linestyle='-',
                                marker='.',
                                alpha=0.3
                                )
                if capacity_plot_mode in ['charge', 'both']:
                    axs[1].plot(specific_charge_capacity['Cycle'],
                                specific_charge_capacity[f'{data_file_names[counter_var]} ChCap'],
                                label=f'{label_prefix} (Charge)' if capacity_plot_mode == 'both' else label_prefix,
                                linestyle='--',
                                marker='.',
                                alpha=0.3
                                )

        # Mean and stddev plots for Capacity
        for counter_var in range(0, different_batches):
            if capacity_plot_mode in ['discharge', 'both']:
                label_dis = f'{legend_list[counter_var]} (Discharge)' if capacity_plot_mode == 'both' else f'{legend_list[counter_var]}'
                axs[1].errorbar(specific_discharge_capacity['Cycle'],
                                specific_discharge_capacity[
                                    '{} mean discharge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                                specific_discharge_capacity[
                                    '{} stddev discharge capacity'.format(
                                        data_file_names[cell_numeration[counter_var]])],
                                label=label_dis
                                , capsize=2.5
                                , marker=marker_list[counter_var % len(marker_list)]
                                , color=color_list[counter_var % len(color_list)]
                                , markersize=6
                                #, errorevery=10
                                )

            if capacity_plot_mode in ['charge', 'both']:
                label_ch = f'{legend_list[counter_var]} (Charge)' if capacity_plot_mode == 'both' else f'{legend_list[counter_var]}'
                axs[1].errorbar(specific_charge_capacity['Cycle'],
                                specific_charge_capacity[
                                    '{} mean charge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                                specific_charge_capacity[
                                    '{} stddev charge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                                label=label_ch
                                , capsize=2.5
                                , marker=marker_list[counter_var % len(marker_list)]
                                , color=color_list[counter_var % len(color_list)]
                                , linestyle='--'
                                , fillstyle='none'
                                , markersize=6
                                #, errorevery=10
                                )

        # Plots for CE
        for counter_var in range(0, different_batches):
            axs[0].errorbar(coulombic_efficiency['Cycle'],
                            coulombic_efficiency['{} mean'.format(data_file_names[cell_numeration[counter_var]])],
                            coulombic_efficiency['{} stddev'.format(data_file_names[cell_numeration[counter_var]])],
                            capsize=2.5
                            #, errorevery=10
                            , marker=marker_list[counter_var % len(marker_list)]
                            , markersize=6
                            , color=color_list[counter_var % len(color_list)]
                            )

        # Apply capacity plot settings
        axs[1].set_title(capacity_plot_title)
        axs[1].set_xlabel('Cycle')
        axs[1].set_ylabel(capacity_ylabel_text)
        axs[1].legend(fontsize=12, loc=0)
        axs[1].grid()
        if capacity_xmin is not None and capacity_xmax is not None:
            axs[1].set_xlim(capacity_xmin, capacity_xmax)
        if capacity_ymin is not None and capacity_ymax is not None:
            axs[1].set_ylim(capacity_ymin, capacity_ymax)
        #axs[1].autoscale(enable=True, axis='both', tight=True)

        # Apply CE plot settings
        axs[0].set_title(ce_plot_title)
        axs[0].set_ylabel(ce_ylabel_text)
        axs[0].grid()
        axs[0].set_xticklabels([])
        if ce_xmin is not None and ce_xmax is not None:
            axs[0].set_xlim(ce_xmin, ce_xmax)
        if ce_ymin is not None and ce_ymax is not None:
            axs[0].set_ylim(ce_ymin, ce_ymax)
        #axs[0].autoscale(enable=True, axis='both', tight=True)

        plt.tight_layout()

        messagebox.showinfo("Success", "Plotting completed successfully. The graphs will now be displayed.")
        plt.show()

    else:
        fig, ax = plt.subplots(nrows=1, ncols=1)

        # Individual cell plots for Capacity (if selected)
        if plot_individual_cells.lower() == 'yes':
            for counter_var in range(0, len(data_file_names)):
                label_prefix = f'Cell {counter_var + 1}'
                if individual_cell_legend_suffix:
                    label_prefix += f' {individual_cell_legend_suffix}'

                if capacity_plot_mode in ['discharge', 'both']:
                    ax.plot(specific_discharge_capacity['Cycle'],
                            specific_discharge_capacity[f'{data_file_names[counter_var]} DisCap'],
                            label=f'{label_prefix} (Discharge)' if capacity_plot_mode == 'both' else label_prefix,
                            linestyle='-',
                            marker='.',
                            alpha=0.3
                            )
                if capacity_plot_mode in ['charge', 'both']:
                    ax.plot(specific_charge_capacity['Cycle'],
                            specific_charge_capacity[f'{data_file_names[counter_var]} ChCap'],
                            label=f'{label_prefix} (Charge)' if capacity_plot_mode == 'both' else label_prefix,
                            linestyle='--',
                            marker='.',
                            alpha=0.3
                            )

        # Mean and stddev plots for Capacity
        for counter_var in range(0, different_batches):
            if capacity_plot_mode in ['discharge', 'both']:
                label_dis = f'{legend_list[counter_var]} (Discharge)' if capacity_plot_mode == 'both' else f'{legend_list[counter_var]}'
                ax.errorbar(specific_discharge_capacity['Cycle'],
                            specific_discharge_capacity[
                                '{} mean discharge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                            specific_discharge_capacity[
                                '{} stddev discharge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                            label=label_dis
                            , capsize=2.5
                            , marker=marker_list[counter_var % len(marker_list)]
                            , color=color_list[counter_var % len(color_list)]
                            )

            if capacity_plot_mode in ['charge', 'both']:
                label_ch = f'{legend_list[counter_var]} (Charge)' if capacity_plot_mode == 'both' else f'{legend_list[counter_var]}'
                ax.errorbar(specific_charge_capacity['Cycle'],
                            specific_charge_capacity[
                                '{} mean charge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                            specific_charge_capacity[
                                '{} stddev charge capacity'.format(data_file_names[cell_numeration[counter_var]])],
                            label=label_ch
                            , capsize=2.5
                            , marker=marker_list[counter_var % len(marker_list)]
                            , color=color_list[counter_var % len(color_list)]
                            , linestyle='--'
                            , fillstyle='none'
                            )

        # Apply capacity plot settings
        ax.set_title(capacity_plot_title)
        ax.set_xlabel('Cycle')
        ax.set_ylabel(capacity_ylabel_text)
        ax.legend(fontsize=11, loc=0)
        ax.grid()
        if capacity_xmin is not None and capacity_xmax is not None:
            ax.set_xlim(capacity_xmin, capacity_xmax)
        if capacity_ymin is not None and capacity_ymax is not None:
            ax.set_ylim(capacity_ymin, capacity_ymax)
        #ax.autoscale(enable=True, axis='both', tight=True)

        plt.tight_layout()
        messagebox.showinfo("Success", "Plotting completed successfully. The graphs will now be displayed.")
        plt.show()

    # Save Figures and Data (as in original script, but now in the function)
    try:
        #plot_name = Path(data_path).parts[-1]
        plot_name = 'cap-vs-cycle_' + str(data_path.parts[-2])
        parent_dir = Path(data_path).parent

        plt.savefig(os.path.join(str(parent_dir), plot_name))
        plt.savefig(os.path.join(str(parent_dir), str(f'{plot_name}.svg')))
        plt.savefig(os.path.join(str(parent_dir), str(f'{plot_name}.pdf')))
        pickle.dump(fig, open((os.path.join(str(parent_dir), plot_name) + '.pickle'), 'wb'))

        save_file_name = os.path.join(str(parent_dir), str(f'{plot_name}.txt'))

        df_to_save = pd.DataFrame({'Cycle': np.arange(1, max_len + 1)})

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

        df_to_save.to_csv(save_file_name, sep=',', index=False, na_rep='')
        print(f"Data has been saved to {save_file_name}.")

    except Exception as e:
        messagebox.showerror("Error while Saving", f"An error occurred while saving the results: {e}")
        return


# --- GUI Application Class ---
class PlottingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advanced Plotting Tool")
        self.geometry("1000x800")

        self.settings_file = "last_used_settings.pkl"
        self.default_dict_file = "dictionary_HIPOLE.txt"
        self.default_data_path = r"C:\Users\ro45vij\Desktop\AA_Data-Processing\AA_Plotting\E5_Vergleich_Swagelok-CC\data"

        self.create_widgets()
        self.load_settings()

    def create_widgets(self):
        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # Create buttons and status label first, so they appear at top
        button_frame = tk.Frame(self)
        button_frame.pack(side="top", pady=10)  # <--- move above notebook
        tk.Button(button_frame, text="Restore Last Settings",
                  command=self.load_settings, font=("Arial", 10)).pack(side='left', padx=5)
        tk.Button(button_frame, text="Run Plotting", command=self.on_run_click,
                  font=("Arial", 12, "bold"), bg="lightblue", relief="raised", padx=10, pady=5).pack(side='left',
                                                                                                     padx=5)

        self.status_label = tk.Label(self, text="", bd=1, relief="sunken", anchor="w")
        self.status_label.pack(side="bottom", fill="x")

        # Create a notebook (tabbed interface)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # Tab 1: Main Inputs
        self.main_frame = tk.Frame(self.notebook, padx=10, pady=10)
        self.notebook.add(self.main_frame, text="Main Settings")
        self.create_main_inputs(self.main_frame)

        # Tab 2: Plot Customization
        self.plot_frame = tk.Frame(self.notebook, padx=10, pady=10)
        self.notebook.add(self.plot_frame, text="Plot Customization")
        self.create_plot_customization(self.plot_frame)

        # Tab 3: Dictionary Editor
        self.dict_frame = tk.Frame(self.notebook, padx=10, pady=10)
        self.notebook.add(self.dict_frame, text="Edit Dictionary")
        self.create_dict_editor(self.dict_frame)

    def create_main_inputs(self, frame):
        labels = ["Data Directory:", "Dictionary File:", "Number of Batches:",
                  "Cells per Batch (comma-separated):"]
        self.entries = {}
        for i, text in enumerate(labels):
            tk.Label(frame, text=text, anchor="w", font=("Arial", 10)).grid(row=i, column=0, sticky="w", pady=5)
            entry = tk.Entry(frame, width=60)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[text] = entry

        self.entries["Data Directory:"].insert(0, self.default_data_path)
        self.entries["Dictionary File:"].insert(0, self.default_dict_file)
        self.entries["Number of Batches:"].insert(0, "4")
        self.entries["Cells per Batch (comma-separated):"].insert(0,
                                                                  "1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2")

        # Browse buttons
        tk.Button(frame, text="Browse", command=self.browse_data_path).grid(row=0, column=2, padx=5)
        tk.Button(frame, text="Browse", command=self.browse_dictionary_file).grid(row=1, column=2, padx=5)

        # Radiobuttons for plot options
        tk.Label(frame, text="CE Graph:", anchor="w", font=("Arial", 10)).grid(row=4, column=0, sticky="w", pady=5)
        self.ce_graph_var = tk.StringVar(value="Yes")
        tk.Radiobutton(frame, text="Yes", variable=self.ce_graph_var, value="Yes").grid(row=4, column=1, sticky="w")
        tk.Radiobutton(frame, text="No", variable=self.ce_graph_var, value="No").grid(row=4, column=1, padx=40,
                                                                                      sticky="w")

        tk.Label(frame, text="Capacity Plot Mode:", anchor="w", font=("Arial", 10)).grid(row=5, column=0, sticky="w",
                                                                                         pady=5)
        self.capacity_plot_mode_var = tk.StringVar(value="both")
        tk.Radiobutton(frame, text="Discharge", variable=self.capacity_plot_mode_var, value="discharge").grid(row=5,
                                                                                                              column=1,
                                                                                                              sticky="w")
        tk.Radiobutton(frame, text="Charge", variable=self.capacity_plot_mode_var, value="charge").grid(row=5, column=1,
                                                                                                        padx=80,
                                                                                                        sticky="w")
        tk.Radiobutton(frame, text="Both", variable=self.capacity_plot_mode_var, value="both").grid(row=5, column=1,
                                                                                                    padx=160,
                                                                                                    sticky="w")

        tk.Label(frame, text="First Cycle Discharge Only:", anchor="w", font=("Arial", 10)).grid(row=6, column=0,
                                                                                                 sticky="w", pady=5)
        self.first_cycle_discharge_var = tk.StringVar(value="Yes")
        tk.Radiobutton(frame, text="Yes", variable=self.first_cycle_discharge_var, value="Yes").grid(row=6, column=1,
                                                                                                     sticky="w")
        tk.Radiobutton(frame, text="No", variable=self.first_cycle_discharge_var, value="No").grid(row=6, column=1,
                                                                                                   padx=40, sticky="w")

        tk.Label(frame, text="Plot Individual Cells:", anchor="w", font=("Arial", 10)).grid(row=7, column=0, sticky="w",
                                                                                            pady=5)
        self.plot_individual_cells_var = tk.StringVar(value="No")
        tk.Radiobutton(frame, text="Yes", variable=self.plot_individual_cells_var, value="Yes").grid(row=7, column=1,
                                                                                                     sticky="w")
        tk.Radiobutton(frame, text="No", variable=self.plot_individual_cells_var, value="No").grid(row=7, column=1,
                                                                                                   padx=40, sticky="w")

    def create_plot_customization(self, frame):
        # General Plot Settings
        tk.Label(frame, text="General Plot Settings", font=("Arial", 12, "bold")).pack(fill='x', pady=(10, 5))

        tk.Label(frame, text="Color List (comma-separated):", anchor="w").pack(fill='x', pady=(5, 0))
        self.color_text = tk.Text(frame, height=4, width=80)
        self.color_text.pack(pady=5)
        self.color_text.insert(tk.END,
                               "tab:blue, tab:orange, tab:green, tab:red, tab:purple, tab:brown, tab:pink, tab:gray, tab:olive, tab:cyan")

        tk.Label(frame, text="Marker List (comma-separated):", anchor="w").pack(fill='x', pady=(5, 0))
        self.marker_text = tk.Text(frame, height=4, width=80)
        self.marker_text.pack(pady=5)
        self.marker_text.insert(tk.END,
                                "o, v, ^, <, >, s, p, 2, 3, 4, 8, s, p, P, *, h, H, +, x, X, D, d, |, _, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11")

        # --- Capacity Plot Settings ---
        tk.Label(frame, text="Capacity Plot Settings", font=("Arial", 12, "bold")).pack(fill='x', pady=(15, 5))

        tk.Label(frame, text="Capacity Plot Title:", anchor="w").pack(fill='x', pady=(5, 0))
        self.capacity_title_entry = tk.Entry(frame, width=80)
        self.capacity_title_entry.pack(pady=5)
        self.capacity_title_entry.insert(0, "Specific Capacity and Coulombic Efficiency")

        tk.Label(frame, text="Capacity Y-axis Label:", anchor="w").pack(fill='x', pady=(5, 0))
        self.capacity_ylabel_entry = tk.Entry(frame, width=80)
        self.capacity_ylabel_entry.pack(pady=5)
        self.capacity_ylabel_entry.insert(0, 'Specific Capacity [mAh $g^{-1}$]')

        tk.Label(frame, text="Individual Cell Legend Suffix:", anchor="w").pack(fill='x', pady=(5, 0))
        self.individual_cell_legend_suffix_entry = tk.Entry(frame, width=80)
        self.individual_cell_legend_suffix_entry.pack(pady=5)
        self.individual_cell_legend_suffix_entry.insert(0, "(Cell)")

        tk.Label(frame, text="Capacity X-axis Limits (min, max):", anchor="w").pack(fill='x', pady=(5, 0))
        cap_xlim_frame = tk.Frame(frame)
        cap_xlim_frame.pack(fill='x')
        self.capacity_xmin_entry = tk.Entry(cap_xlim_frame, width=10)
        self.capacity_xmin_entry.pack(side='left', padx=5)
        tk.Label(cap_xlim_frame, text=",").pack(side='left')
        self.capacity_xmax_entry = tk.Entry(cap_xlim_frame, width=10)
        self.capacity_xmax_entry.pack(side='left', padx=5)

        tk.Label(frame, text="Capacity Y-axis Limits (min, max):", anchor="w").pack(fill='x', pady=(5, 0))
        cap_ylim_frame = tk.Frame(frame)
        cap_ylim_frame.pack(fill='x')
        self.capacity_ymin_entry = tk.Entry(cap_ylim_frame, width=10)
        self.capacity_ymin_entry.pack(side='left', padx=5)
        tk.Label(cap_ylim_frame, text=",").pack(side='left')
        self.capacity_ymax_entry = tk.Entry(cap_ylim_frame, width=10)
        self.capacity_ymax_entry.pack(side='left', padx=5)
        self.capacity_ymax_entry.insert(0, "210")

        # --- CE Plot Settings ---
        tk.Label(frame, text="CE Plot Settings (only used if 'CE Graph' is 'Yes')", font=("Arial", 12, "bold")).pack(
            fill='x', pady=(15, 5))

        tk.Label(frame, text="CE Plot Title:", anchor="w").pack(fill='x', pady=(5, 0))
        self.ce_title_entry = tk.Entry(frame, width=80)
        self.ce_title_entry.pack(pady=5)
        self.ce_title_entry.insert(0, "Coulombic Efficiency")

        tk.Label(frame, text="CE Y-axis Label:", anchor="w").pack(fill='x', pady=(5, 0))
        self.ce_ylabel_entry = tk.Entry(frame, width=80)
        self.ce_ylabel_entry.pack(pady=5)
        self.ce_ylabel_entry.insert(0, 'CE [%]')

        tk.Label(frame, text="CE X-axis Limits (min, max):", anchor="w").pack(fill='x', pady=(5, 0))
        ce_xlim_frame = tk.Frame(frame)
        ce_xlim_frame.pack(fill='x')
        self.ce_xmin_entry = tk.Entry(ce_xlim_frame, width=10)
        self.ce_xmin_entry.pack(side='left', padx=5)
        tk.Label(ce_xlim_frame, text=",").pack(side='left')
        self.ce_xmax_entry = tk.Entry(ce_xlim_frame, width=10)
        self.ce_xmax_entry.pack(side='left', padx=5)

        tk.Label(frame, text="CE Y-axis Limits (min, max):", anchor="w").pack(fill='x', pady=(5, 0))
        ce_ylim_frame = tk.Frame(frame)
        ce_ylim_frame.pack(fill='x')
        self.ce_ymin_entry = tk.Entry(ce_ylim_frame, width=10)
        self.ce_ymin_entry.pack(side='left', padx=5)
        self.ce_ymin_entry.insert(0, "98")
        tk.Label(ce_ylim_frame, text=",").pack(side='left')
        self.ce_ymax_entry = tk.Entry(ce_ylim_frame, width=10)
        self.ce_ymax_entry.pack(side='left', padx=5)
        self.ce_ymax_entry.insert(0, "102")

    def create_dict_editor(self, frame):
        tk.Label(frame, text="Edit content of dictionary_HIPOLE.txt:", anchor="w").pack(fill='x', pady=(10, 0))
        self.dict_text = tk.Text(frame, height=20, width=80)
        self.dict_text.pack(pady=5, expand=True, fill='both')

        dict_button_frame = tk.Frame(frame)
        dict_button_frame.pack(pady=5)
        tk.Button(dict_button_frame, text="Load Dictionary", command=self.load_dictionary).pack(side='left', padx=5)
        tk.Button(dict_button_frame, text="Save Dictionary", command=self.save_dictionary).pack(side='left', padx=5)

        self.load_dictionary()  # Load the dictionary on startup

    def browse_data_path(self):
        directory = filedialog.askdirectory(title="Select Data Directory")
        if directory:
            self.entries["Data Directory:"].delete(0, tk.END)
            self.entries["Data Directory:"].insert(0, directory)

    def browse_dictionary_file(self):
        file_path = filedialog.askopenfilename(title="Select Dictionary File", filetypes=[("Text files", "*.txt")])
        if file_path:
            self.entries["Dictionary File:"].delete(0, tk.END)
            self.entries["Dictionary File:"].insert(0, file_path)
            self.load_dictionary()

    def load_dictionary(self):
        dict_path = self.entries["Dictionary File:"].get()
        if not Path(dict_path).is_file():
            self.dict_text.delete(1.0, tk.END)
            self.status_label.config(text="Warning: Dictionary file not found.")
            return

        try:
            with open(dict_path, "r") as file:
                content = file.read()
                self.dict_text.delete(1.0, tk.END)
                self.dict_text.insert(tk.END, content)
                self.status_label.config(text="Dictionary loaded successfully.")
        except Exception as e:
            messagebox.showerror("Loading Error", f"Could not load dictionary: {e}")
            self.status_label.config(text="Error loading dictionary.")

    def save_dictionary(self):
        dict_path = self.entries["Dictionary File:"].get()
        content = self.dict_text.get(1.0, tk.END)
        try:
            # Validate content before saving
            ast.literal_eval(content)
            with open(dict_path, "w") as file:
                file.write(content)
            self.status_label.config(text="Dictionary saved successfully.")
        except (SyntaxError, ValueError) as e:
            messagebox.showerror("Save Error", f"Invalid dictionary format. Please check it. Error: {e}")
            self.status_label.config(text="Error: Invalid dictionary format.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save dictionary: {e}")
            self.status_label.config(text="Error saving dictionary.")

    def save_settings(self):
        settings = {
            "data_path": self.entries["Data Directory:"].get(),
            "dictionary_name": self.entries["Dictionary File:"].get(),
            "different_batches": self.entries["Number of Batches:"].get(),
            "number_of_cells": self.entries["Cells per Batch (comma-separated):"].get(),
            "ce_graph": self.ce_graph_var.get(),
            "capacity_plot_mode": self.capacity_plot_mode_var.get(),
            "first_cycle_discharge_only": self.first_cycle_discharge_var.get(),
            "plot_individual_cells": self.plot_individual_cells_var.get(),
            "color_list": self.color_text.get(1.0, tk.END).strip(),
            "marker_list": self.marker_text.get(1.0, tk.END).strip(),
            "capacity_plot_title": self.capacity_title_entry.get(),
            "capacity_ylabel_text": self.capacity_ylabel_entry.get(),
            "individual_cell_legend_suffix": self.individual_cell_legend_suffix_entry.get(),
            "ce_plot_title": self.ce_title_entry.get(),
            "ce_ylabel_text": self.ce_ylabel_entry.get(),
            "ce_ymin": self.ce_ymin_entry.get(),
            "ce_ymax": self.ce_ymax_entry.get(),
            "capacity_xmin": self.capacity_xmin_entry.get(),
            "capacity_xmax": self.capacity_xmax_entry.get(),
            "capacity_ymin": self.capacity_ymin_entry.get(),
            "capacity_ymax": self.capacity_ymax_entry.get(),
            "ce_xmin": self.ce_xmin_entry.get(),
            "ce_xmax": self.ce_xmax_entry.get()
        }
        try:
            with open(self.settings_file, 'wb') as f:
                pickle.dump(settings, f)
            self.status_label.config(text="Settings saved.")
        except Exception as e:
            self.status_label.config(text=f"Error saving settings: {e}")

    def load_settings(self):
        if not Path(self.settings_file).is_file():
            self.status_label.config(text="No saved settings found.")
            return

        try:
            with open(self.settings_file, 'rb') as f:
                settings = pickle.load(f)

            self.entries["Data Directory:"].delete(0, tk.END)
            self.entries["Data Directory:"].insert(0, settings.get("data_path", ""))
            self.entries["Dictionary File:"].delete(0, tk.END)
            self.entries["Dictionary File:"].insert(0, settings.get("dictionary_name", ""))
            self.entries["Number of Batches:"].delete(0, tk.END)
            self.entries["Number of Batches:"].insert(0, settings.get("different_batches", ""))
            self.entries["Cells per Batch (comma-separated):"].delete(0, tk.END)
            self.entries["Cells per Batch (comma-separated):"].insert(0, settings.get("number_of_cells", ""))

            self.ce_graph_var.set(settings.get("ce_graph", "Yes"))
            self.capacity_plot_mode_var.set(settings.get("capacity_plot_mode", "both"))
            self.first_cycle_discharge_var.set(settings.get("first_cycle_discharge_only", "Yes"))
            self.plot_individual_cells_var.set(settings.get("plot_individual_cells", "No"))

            self.color_text.delete(1.0, tk.END)
            self.color_text.insert(tk.END, settings.get("color_list", ""))
            self.marker_text.delete(1.0, tk.END)
            self.marker_text.insert(tk.END, settings.get("marker_list", ""))

            self.capacity_title_entry.delete(0, tk.END)
            self.capacity_title_entry.insert(0, settings.get("capacity_plot_title", ""))
            self.capacity_ylabel_entry.delete(0, tk.END)
            self.capacity_ylabel_entry.insert(0, settings.get("capacity_ylabel_text", ""))
            self.individual_cell_legend_suffix_entry.delete(0, tk.END)
            self.individual_cell_legend_suffix_entry.insert(0, settings.get("individual_cell_legend_suffix", "(Cell)"))

            self.capacity_xmin_entry.delete(0, tk.END)
            self.capacity_xmin_entry.insert(0, settings.get("capacity_xmin", ""))
            self.capacity_xmax_entry.delete(0, tk.END)
            self.capacity_xmax_entry.insert(0, settings.get("capacity_xmax", ""))
            self.capacity_ymin_entry.delete(0, tk.END)
            self.capacity_ymin_entry.insert(0, settings.get("capacity_ymin", ""))
            self.capacity_ymax_entry.delete(0, tk.END)
            self.capacity_ymax_entry.insert(0, settings.get("capacity_ymax", ""))

            self.ce_title_entry.delete(0, tk.END)
            self.ce_title_entry.insert(0, settings.get("ce_plot_title", ""))
            self.ce_ylabel_entry.delete(0, tk.END)
            self.ce_ylabel_entry.insert(0, settings.get("ce_ylabel_text", ""))
            self.ce_xmin_entry.delete(0, tk.END)
            self.ce_xmin_entry.insert(0, settings.get("ce_xmin", ""))
            self.ce_xmax_entry.delete(0, tk.END)
            self.ce_xmax_entry.insert(0, settings.get("ce_xmax", ""))
            self.ce_ymin_entry.delete(0, tk.END)
            self.ce_ymin_entry.insert(0, settings.get("ce_ymin", ""))
            self.ce_ymax_entry.delete(0, tk.END)
            self.ce_ymax_entry.insert(0, settings.get("ce_ymax", ""))

            self.status_label.config(text="Settings loaded successfully.")
        except Exception as e:
            self.status_label.config(text=f"Error loading settings: {e}")

    def on_run_click(self):
        # Save settings before running the plot
        self.save_settings()

        # Get all values from the GUI
        data_path_str = self.entries["Data Directory:"].get()
        dictionary_name_str = self.entries["Dictionary File:"].get()
        different_batches_str = self.entries["Number of Batches:"].get()
        number_of_cells_str = self.entries["Cells per Batch (comma-separated):"].get()
        ce_graph_str = self.ce_graph_var.get()
        capacity_plot_mode_str = self.capacity_plot_mode_var.get()
        first_cycle_discharge_only_str = self.first_cycle_discharge_var.get()
        plot_individual_cells_str = self.plot_individual_cells_var.get()
        color_list_str = self.color_text.get(1.0, tk.END).strip()
        marker_list_str = self.marker_text.get(1.0, tk.END).strip()
        capacity_plot_title_str = self.capacity_title_entry.get()
        capacity_ylabel_text_str = self.capacity_ylabel_entry.get()
        individual_cell_legend_suffix_str = self.individual_cell_legend_suffix_entry.get()
        ce_plot_title_str = self.ce_title_entry.get()
        ce_ylabel_text_str = self.ce_ylabel_entry.get()
        capacity_xmin_str = self.capacity_xmin_entry.get()
        capacity_xmax_str = self.capacity_xmax_entry.get()
        capacity_ymin_str = self.capacity_ymin_entry.get()
        capacity_ymax_str = self.capacity_ymax_entry.get()
        ce_xmin_str = self.ce_xmin_entry.get()
        ce_xmax_str = self.ce_xmax_entry.get()
        ce_ymin_str = self.ce_ymin_entry.get()
        ce_ymax_str = self.ce_ymax_entry.get()

        run_plotting(data_path_str, dictionary_name_str, different_batches_str, number_of_cells_str,
                     ce_graph_str, capacity_plot_mode_str, first_cycle_discharge_only_str, plot_individual_cells_str,
                     color_list_str, marker_list_str, capacity_plot_title_str, capacity_ylabel_text_str,
                     capacity_xmin_str, capacity_xmax_str, capacity_ymin_str, capacity_ymax_str,
                     ce_plot_title_str, ce_ylabel_text_str, ce_xmin_str, ce_xmax_str, ce_ymin_str, ce_ymax_str,
                     individual_cell_legend_suffix_str)


if __name__ == '__main__':
    app = PlottingGUI()
    app.mainloop()
