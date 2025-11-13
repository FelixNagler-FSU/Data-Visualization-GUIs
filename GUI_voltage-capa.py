### authors: Felix Nagler
### v0.1, Nov 2025

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
import zipfile
import xarray as xr
import shutil

# Use TkAgg backend to integrate with the tkinter GUI
# This is required to show the matplotlib plot within a tkinter application.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors


# --- Refactored plotting logic into a function ---
def run_plotting(
    data_path_str,
    dictionary_name_str,
    cycle_numbers_str,
    plot_title_str,
    xlabel_text_str,
    ylabel_text_str,
    color_list_str,
    alpha_list_str,
    xmin_str,
    xmax_str,
    ymin_str,
    ymax_str,
    first_cycle_discharge_only,
    close_app_on_plot_close=False,
    # Save options
    save_png=True,
    save_pdf=True,
    save_svg=True,
    save_txt=True,
    save_netcdf=True,
    save_zip=True,
):
    """Core logic for voltage vs specific capacity plotting.

    High-level flow:
    1. Parse & validate inputs (paths, numeric limits, cycle list).
    2. Read .mpr files (skip non-.mpr). Extract voltage (Ewe/Ecell) and capacity (Q/Capacity).
    3. Split data by half-cycles; map half-cycles to full cycles (conditional first-cycle discharge-only logic).
    4. Build per-cycle charge/discharge capacity & voltage series (last sample per half-cycle retained).
    5. Assemble DataFrames per cycle for plotting.
    6. Generate matplotlib figure: overlay charge/discharge curves across selected cycles.
    7. Save figure (PNG/SVG/PDF), pickle, TXT export of flattened data.
    8. Export per-cell raw arrays to NetCDF (unique dimensions for differing lengths) and ZIP them.

    Key conditions and their outcomes:
    - Missing directory / dictionary file -> abort with messagebox error.
    - Non-convertible numeric inputs -> abort with messagebox error.
    - Non-.mpr files -> skipped (printed).
    - Columns normalization: rename 'Ecell'->'Ewe', 'Capacity'->'Q' if needed.
    - First cycle discharge-only flag True -> treat half-cycle 0 as cycle 1 discharge, adjust mapping.
    - Axis limits applied only if both min & max given (None keeps autoscale).
    - Variable length arrays across cycles -> separate point dimension per exported NetCDF variable.

    Args:
        data_path_str (str): Directory containing raw .mpr files.
        dictionary_name_str (str): Path to legend mapping text file.
        cycle_numbers_str (str): Comma-separated cycles to plot (e.g. "1, 5, 10").
        plot_title_str (str): Figure title.
        xlabel_text_str (str): X-axis label (usually capacity).
        ylabel_text_str (str): Y-axis label (usually voltage).
        color_list_str (str): Comma-separated matplotlib color spec list.
        alpha_list_str (str): Comma-separated floats for transparency cycling per cycle.
        xmin_str/xmax_str/ymin_str/ymax_str (str): Optional axis bounds.
        first_cycle_discharge_only (bool): Special mapping rule for first cycle.
        close_app_on_plot_close (bool): If True attempt to close Tk root after plotting.
    Returns:
        None. Side effects: files written, plot shown (non-blocking), NetCDF + ZIP saved.
    """
    # Convert string inputs to the required data types for processing
    try:
        data_path = Path(data_path_str)
        dictionary_name = Path(dictionary_name_str)
        # Parse the comma-separated string into a list of integers
        cycle_numbers = [int(x.strip()) for x in cycle_numbers_str.split(',')]
        plot_title = plot_title_str
        xlabel_text = xlabel_text_str
        ylabel_text = ylabel_text_str
        # Parse comma-separated color and alpha strings into lists
        color_list = [c.strip() for c in color_list_str.split(',')]
        alpha_list = [float(a.strip()) for a in alpha_list_str.split(',')]

        # Parse and handle optional numerical inputs for plot limits
        xmin = float(xmin_str) if xmin_str else None
        xmax = float(xmax_str) if xmax_str else None
        ymin = float(ymin_str) if ymin_str else None
        ymax = float(ymax_str) if ymax_str else None

    except (ValueError, IndexError) as e:
        # Show an error message if any input conversion fails
        messagebox.showerror("Input Error", f"Please check your inputs. Error: {e}")
        return

    # Check if the directories and dictionary file exist
    if not data_path.is_dir():
        messagebox.showerror("Error", f"Data directory not found: {data_path}")
        return
    if not dictionary_name.is_file():
        messagebox.showerror("Error", f"Dictionary file not found: {dictionary_name}")
        return

    # List all data files in the specified directory
    data_file_names = os.listdir(data_path)

    # Initialize dictionaries to store processed data and active material weights
    max_rows = 0
    all_data = {cycle: {} for cycle in cycle_numbers}
    weight_dict = {}

    # Main loop to read and process each data file
    for idx, filename in enumerate(data_file_names):
        file_path = data_path / filename

        df = None
        suffix = Path(filename).suffix.lower()

        # Try .mpr via yadg first
        if suffix == '.mpr':
            try:
                import yadg
                import json
            except Exception:
                print(f"yadg not available — falling back to header parsing for {filename}.")
            else:
                try:
                    ds_raw = yadg.extractors.extract(filetype='eclab.mpr', path=str(file_path))
                except Exception as e:
                    print(f"Warning: yadg failed to parse {filename}: {e}. Falling back to header parsing.")
                else:
                    # Extract active material mass from original_metadata if present
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
                        print(f"Warning: No active material mass found in {filename}. Skipping file.")
                    else:
                        # Convert datatree -> xarray Dataset if needed
                        try:
                            if hasattr(ds_raw, 'to_dataset'):
                                ds = ds_raw.to_dataset()
                            elif hasattr(ds_raw, 'to_xarray'):
                                ds = ds_raw.to_xarray()
                            else:
                                if hasattr(ds_raw, 'get') and ds_raw.get('/') is not None:
                                    ds = ds_raw.get('/').to_dataset()
                                else:
                                    ds = ds_raw
                        except Exception:
                            ds = ds_raw

                        # Find Q (per-sample capacity-like) and Ewe (voltage) and half-cycle
                        keys = list(ds.keys()) if hasattr(ds, 'keys') else list(getattr(ds, 'data_vars', {}))
                        q_var = next((k for k in keys if 'Q charge or discharge' in k or 'Q charge/discharge' in k or 'Q' == k), None)
                        e_var = next((k for k in keys if any(sub in k for sub in ('Ewe', 'Ecell', 'Ecell/V', 'Ewe/V'))), None)
                        half_var = next((k for k in keys if any(sub in k for sub in ('half cycle', 'Half_cycle', 'Half cycle', 'half_cycle'))), None)

                        if q_var is None or e_var is None or half_var is None:
                            print(f"Warning: Could not find Q/Ewe/Half_cycle variables in {filename}. Skipping.")
                        else:
                            try:
                                q_arr = np.asarray(ds[q_var].values) if hasattr(ds[q_var], 'values') else np.asarray(ds[q_var])
                                e_arr = np.asarray(ds[e_var].values) if hasattr(ds[e_var], 'values') else np.asarray(ds[e_var])
                                half_arr = np.asarray(ds[half_var].values) if hasattr(ds[half_var], 'values') else np.asarray(ds[half_var])

                                df = pd.DataFrame({'Q': q_arr, 'Ewe': e_arr, 'Half_cycle': half_arr})
                            except Exception as e:
                                print(f"Warning: error creating DataFrame from {filename}: {e}. Skipping.")

        # We only process .mpr files now
        if df is None:
            print(f"Skipping non-mpr file or parsing failed: {filename}")
            continue

        # Ensure we have the required columns
        if 'Half_cycle' not in df.columns:
            print(f"Skipping {filename}: Half_cycle column not found after parsing.")
            continue

        if 'Ewe' not in df.columns and 'Ecell' in df.columns:
            df.rename(columns={'Ecell': 'Ewe'}, inplace=True)
        if 'Q' not in df.columns and 'Capacity' in df.columns:
            df.rename(columns={'Capacity': 'Q'}, inplace=True)

        if df.shape[0] > max_rows:
            max_rows = df.shape[0]

        # Group data by 'Half_cycle' to separate charge and discharge phases
        grouped = df.groupby('Half_cycle')
        temp_data_dict = {}

        # Determine mass (weight) for normalization; default to 1 if unknown
        weight = weight_dict.get(filename, None)
        if weight is None:
            weight = 1.0

        for half_cycle_value, group in grouped:
            if group.empty:
                continue

            # Logic to determine charge/discharge and assign cycle number
            try:
                h_val = int(half_cycle_value)
            except Exception:
                # sometimes half-cycle is string; try to cast
                try:
                    h_val = int(float(half_cycle_value))
                except Exception:
                    continue

            is_charge = False
            if first_cycle_discharge_only:
                if h_val == 0:
                    cycle_num = 1
                    is_charge = False
                else:
                    cycle_num = (h_val - 2) // 2 + 2
                    is_charge = (h_val % 2 == 0)
            else:
                cycle_num = h_val // 2 + 1
                is_charge = (h_val % 2 == 0)

            if is_charge:
                volt_col = f'chVolt{cycle_num}'
                cap_col = f'chCap{cycle_num}'
            else:
                volt_col = f'disVolt{cycle_num}'
                cap_col = f'disCap{cycle_num}'

            # For .mpr we used 'Q' and 'Ewe'; for legacy header we may have had 'Capacity' and 'Ewe'
            volt_series = group.get('Ewe') if 'Ewe' in group else group.get('Ecell')
            cap_series = group.get('Q') if 'Q' in group else group.get('Capacity') if 'Capacity' in group else None

            if volt_series is None or cap_series is None:
                # skip this half if necessary data missing
                continue

            # Normalize capacity by active material weight (mass in g). If mass was not found, weight==1.
            try:
                cap_norm = cap_series.reset_index(drop=True).astype(float) / float(weight)
            except Exception:
                cap_norm = cap_series.reset_index(drop=True)

            temp_data_dict[volt_col] = volt_series.reset_index(drop=True)
            temp_data_dict[cap_col] = cap_norm

        # Create a DataFrame from the processed cycle data
        cycle_separated_df = pd.DataFrame(temp_data_dict)

        # Populate the main data dictionary for plotting
        for cycle in cycle_numbers:
            chCap_col = f'chCap{cycle}'
            chVolt_col = f'chVolt{cycle}'
            disCap_col = f'disCap{cycle}'
            disVolt_col = f'disVolt{cycle}'

            if chCap_col in cycle_separated_df.columns and chVolt_col in cycle_separated_df.columns:
                all_data[cycle][f'{filename}_ch_cap'] = cycle_separated_df[chCap_col]
                all_data[cycle][f'{filename}_ch_volt'] = cycle_separated_df[chVolt_col]

            if disCap_col in cycle_separated_df.columns and disVolt_col in cycle_separated_df.columns:
                # Take absolute value of discharge capacity
                all_data[cycle][f'{filename}_dis_cap'] = cycle_separated_df[disCap_col].abs()
                all_data[cycle][f'{filename}_dis_volt'] = cycle_separated_df[disVolt_col]

    # Create separate DataFrames for charge and discharge data for easier plotting
    capacity_charge_all = {cycle: pd.DataFrame({k: v for k, v in all_data[cycle].items() if '_ch_cap' in k}) for cycle
                           in cycle_numbers}
    voltage_charge_all = {cycle: pd.DataFrame({k: v for k, v in all_data[cycle].items() if '_ch_volt' in k}) for cycle
                          in cycle_numbers}
    capacity_discharge_all = {cycle: pd.DataFrame({k: v for k, v in all_data[cycle].items() if '_dis_cap' in k}) for
                              cycle in cycle_numbers}
    voltage_discharge_all = {cycle: pd.DataFrame({k: v for k, v in all_data[cycle].items() if '_dis_volt' in k}) for
                             cycle in cycle_numbers}

    for cycle in capacity_charge_all:
        # Replace zero values with NaN to prevent plotting them
        capacity_charge_all[cycle].replace(0, np.nan, inplace=True)

    # Read the legend dictionary from the provided text file
    try:
        with open(dictionary_name, "r", encoding="cp1252", errors='ignore') as file:
            contents = file.read()
            # Safely evaluate the string as a Python dictionary
            dic_legend_list = ast.literal_eval(contents)
    except FileNotFoundError:
        messagebox.showerror("Error", f"Dictionary file not found: {dictionary_name}")
        return
    except (SyntaxError, ValueError) as e:
        messagebox.showerror("Error", f"Invalid format in dictionary file. Error: {e}")
        return

    legend_list = []
    # Create a filtered list of data files that have associated weights
    filtered_data_file_names = [f for f in data_file_names if f in weight_dict]
    # Map the filenames to the legend names using the dictionary
    for filename in filtered_data_file_names:
        found = False
        for key in dic_legend_list.keys():
            if key in filename:
                legend_list.append(str(dic_legend_list[key][0]))
                found = True
                break
        if not found:
            messagebox.showerror("Error", "Not all batches were found in dictionary")
            return

    # --- Plotting execution ---
    fig, ax1 = plt.subplots(nrows=1, ncols=1)

    # Loop through each cycle and file to plot the data
    for i, cycle in enumerate(cycle_numbers):
        for col_idx, filename in enumerate(filtered_data_file_names):
            ch_cap_col = f'{filename}_ch_cap'
            ch_volt_col = f'{filename}_ch_volt'

            dis_cap_col = f'{filename}_dis_cap'
            dis_volt_col = f'{filename}_dis_volt'

            # Get colors and alpha values from the user-provided lists
            color_list = [c.strip() for c in color_list_str.split(',')]
            alpha_list = [float(a.strip()) for a in alpha_list_str.split(',')]
            color = color_list[col_idx % len(color_list)]
            alpha = alpha_list[i % len(alpha_list)]

            # Plot charge and discharge curves if data exists
            if ch_cap_col in capacity_charge_all[cycle].columns:
                ax1.plot(
                    capacity_charge_all[cycle][ch_cap_col],
                    voltage_charge_all[cycle][ch_volt_col],
                    color=color,
                    alpha=alpha,

                )
            if dis_cap_col in capacity_discharge_all[cycle].columns:
                ax1.plot(
                    capacity_discharge_all[cycle][dis_cap_col],
                    voltage_discharge_all[cycle][dis_volt_col],
                    color=color,
                    alpha=alpha,
                    label=f'{legend_list[col_idx]} (Cycle {cycle})'
                )

    # Apply plot settings from user input
    ax1.set_xlabel(xlabel_text)
    ax1.set_ylabel(ylabel_text)
    ax1.legend(fontsize=10)
    ax1.grid()
    ax1.set_title(plot_title)

    # Set axis limits if provided
    if xmin is not None and xmax is not None:
        ax1.set_xlim(xmin, xmax)
    if ymin is not None and ymax is not None:
        ax1.set_ylim(ymin, ymax)

    plt.tight_layout()

    # Define the save directory as the parent of the data directory
    save_dir = data_path.parent

    # Save Figures and data BEFORE showing the plot (in case user closes/kills process)
    try:
        # Include both the experiment folder (parent of "data") and the data folder name in the filename
        # Example base: volt-vs-cap_[25]Experiment-data
        plot_filename_base = f"volt-vs-cap_[{cycle_numbers_str}]{data_path.parts[-2]}-{data_path.parts[-1]}"
        plot_path = os.path.join(save_dir, plot_filename_base)

        if save_png:
            plt.savefig(f'{plot_path}.png')
        if save_svg:
            plt.savefig(f'{plot_path}.svg')
        if save_pdf:
            plt.savefig(f'{plot_path}.pdf')
        pickle.dump(fig, open(f'{plot_path}.pickle', 'wb'))
        print(f"Figures have been saved to {save_dir}.")

    except Exception as e:
        messagebox.showerror("Error while Saving", f"An error occurred while saving the results: {e}")
        return

    # --- NEW: Save all plotted data to a text file ---
    try:
        # Create a single DataFrame to hold all the plotted data
        all_cycles_data = pd.DataFrame()
        for cycle in cycle_numbers:
            # Combine charge and discharge data for the current cycle
            cycle_charge_data = pd.DataFrame({
                f'Cycle_{cycle}_Charge_Capacity': capacity_charge_all[cycle].stack(),
                f'Cycle_{cycle}_Charge_Voltage': voltage_charge_all[cycle].stack()
            }).reset_index(drop=True)

            cycle_discharge_data = pd.DataFrame({
                f'Cycle_{cycle}_Discharge_Capacity': capacity_discharge_all[cycle].stack(),
                f'Cycle_{cycle}_Discharge_Voltage': voltage_discharge_all[cycle].stack()
            }).reset_index(drop=True)

            # Join the dataframes. This will align them based on their row indices.
            cycle_data = pd.concat([cycle_charge_data, cycle_discharge_data], axis=1)

            # Add this cycle's data to the master DataFrame
            if all_cycles_data.empty:
                all_cycles_data = cycle_data
            else:
                all_cycles_data = pd.concat([all_cycles_data, cycle_data], axis=1)

        # Save the master DataFrame to a tab-separated text file
        if save_txt:
            data_save_path = os.path.join(save_dir, f'{plot_filename_base}.txt')
            all_cycles_data.to_csv(data_save_path, sep='\t', index=False, float_format='%.4f')
            print(f"Plotted data saved to {data_save_path}")

        # Decide NetCDF output strategy
        temp_nc_dir = None
        permanent_nc_dir = None
        if save_netcdf or save_zip:
            if save_zip and not save_netcdf:
                temp_nc_dir = os.path.join(save_dir, f"{plot_filename_base}_nc_temp")
                os.makedirs(temp_nc_dir, exist_ok=True)
            elif save_netcdf and not save_zip:
                permanent_nc_dir = os.path.join(save_dir, f"{plot_filename_base}_nc")
                os.makedirs(permanent_nc_dir, exist_ok=True)
            else:
                temp_nc_dir = os.path.join(save_dir, f"{plot_filename_base}_nc_temp")
                os.makedirs(temp_nc_dir, exist_ok=True)

        # Group files by batch using the legend_list (only if exporting NetCDF)
        batch_files = {}
        if save_netcdf or save_zip:
            print("\nProcessing files for NetCDF export:")
            print(f"Number of files to process: {len(filtered_data_file_names)}")
            for idx, filename in enumerate(filtered_data_file_names):
                batch_legend = legend_list[idx]
                print(f"Processing file {filename} with legend {batch_legend}")
                if batch_legend not in batch_files:
                    batch_files[batch_legend] = []
                batch_files[batch_legend].append(filename)

        # Save each cell as a separate NetCDF file using the legend name
        nc_output_dir = permanent_nc_dir if permanent_nc_dir else temp_nc_dir
        saved_nc_files = []
        for batch_name, files in batch_files.items():
            for cell_idx, filename in enumerate(files, 1):
                # Create dataset for this cell
                cell_data = {}
                
                # Get raw data for this cell
                for cycle in cycle_numbers:
                    ch_cap = capacity_charge_all[cycle].get(f'{filename}_ch_cap')
                    ch_volt = voltage_charge_all[cycle].get(f'{filename}_ch_volt')
                    dis_cap = capacity_discharge_all[cycle].get(f'{filename}_dis_cap')
                    dis_volt = voltage_discharge_all[cycle].get(f'{filename}_dis_volt')
                    
                    if ch_cap is not None and ch_volt is not None:
                        cell_data[f'cycle_{cycle}_charge_capacity'] = ch_cap.values
                        cell_data[f'cycle_{cycle}_charge_voltage'] = ch_volt.values
                    if dis_cap is not None and dis_volt is not None:
                        cell_data[f'cycle_{cycle}_discharge_capacity'] = dis_cap.values
                        cell_data[f'cycle_{cycle}_discharge_voltage'] = dis_volt.values

                if cell_data:  # Only create dataset if we have data
                    # Convert to xarray Dataset
                    # Different cycles/branches can have different lengths.
                    # Condition: arrays have unequal sizes across cycles or between charge/discharge.
                    # Outcome: give each variable its own dimension to avoid conflicting sizes in NetCDF.
                    var_dict = {}
                    coords_dict = {}
                    for name, data in cell_data.items():
                        dim_name = f'point_{name}'
                        var_dict[name] = ([dim_name], data)
                        coords_dict[dim_name] = np.arange(len(data))
                    ds = xr.Dataset(var_dict, coords=coords_dict)
                    # Add metadata for traceability and reproducibility
                    ds.attrs['cell_number'] = cell_idx
                    ds.attrs['batch_name'] = batch_name
                    ds.attrs['filename'] = filename
                    if filename in weight_dict:
                        ds.attrs['active_material_mass_mg'] = weight_dict[filename] * 1000  # Convert back to mg
                    
                    # Create safe filename from batch name and cell number
                    safe_batch_name = "".join(c for c in batch_name if c.isalnum() or c in (' ', '-', '_')).strip()
                    nc_filename = f'{safe_batch_name}-Cell{cell_idx}.nc'
                    nc_path = os.path.join(nc_output_dir, nc_filename)
                    
                    print(f"\nSaving NetCDF file for {batch_name} Cell {cell_idx}:")
                    print(f"Path: {nc_path}")
                    try:
                        ds.to_netcdf(nc_path)
                        print("Successfully saved NetCDF file")
                        saved_nc_files.append(nc_path)
                    except Exception as e:
                        print(f"Error saving NetCDF file: {e}")
        # Create ZIP archive if requested
        if (save_netcdf or save_zip) and save_zip and saved_nc_files:
            zip_path = os.path.join(save_dir, f'{plot_filename_base}_raw_data.zip')
            print(f"\nCreating ZIP archive at {zip_path}")
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in saved_nc_files:
                    arcname = os.path.basename(file_path)
                    print(f"Adding file to ZIP: {arcname}")
                    zipf.write(file_path, arcname)
            print(f"Raw data saved as NetCDF files in {zip_path}")
            if temp_nc_dir and os.path.isdir(temp_nc_dir):
                shutil.rmtree(temp_nc_dir)

    except Exception as e:
        messagebox.showerror("Error while Saving Data", f"An error occurred while saving the data: {e}")
        return

    # Show the plot now that everything is saved
    messagebox.showinfo("Success", "Plotting completed successfully. The graph will now be displayed.")
    plt.show(block=False)
    # Don't close the figures immediately - let the user close them manually
    return


# --- GUI Application Class ---
class VoltVsCapGUI(tk.Tk):
    """
    Main class for the Voltage vs. Capacity Plotting GUI application.
    It creates the user interface, handles user interactions, and calls the plotting function.
    """

    def __init__(self):
        # Initialize the main tkinter window
        super().__init__()
        self.title("Voltage vs. Capacity Plotting Tool")
        self.geometry("800x700")
        self.settings_file = "last_volt_cap_settings.pkl"
        self.default_data_path = r"C:\Users\ro45vij\Desktop\AA_Data-Processing\AA_Plotting\E4a+b+c\data"
        self.default_dict_file = "dictionary_HIPOLE.txt"
        self.create_widgets()
        # Load the last saved settings upon startup
        self.load_settings()

    def create_widgets(self):
        """Creates all the GUI widgets and organizes them using a notebook layout."""
        # Status bar at the bottom of the window
        self.status_label = tk.Label(self, text="", bd=1, relief="sunken", anchor="w")
        self.status_label.pack(side="bottom", fill="x")

        # Create a notebook widget with multiple tabs for different settings
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=10, pady=10)

        # Tab for main input settings (data paths, cycles)
        self.main_frame = tk.Frame(self.notebook, padx=10, pady=10)
        self.notebook.add(self.main_frame, text="Main Settings")
        self.create_main_inputs(self.main_frame)

        # Tab for plot customization (title, labels, colors)
        self.plot_frame = tk.Frame(self.notebook, padx=10, pady=10)
        self.notebook.add(self.plot_frame, text="Plot Customization")
        self.create_plot_customization(self.plot_frame)

        # Tab for editing the legend dictionary
        self.dict_frame = tk.Frame(self.notebook, padx=10, pady=10)
        self.notebook.add(self.dict_frame, text="Edit Dictionary")
        self.create_dict_editor(self.dict_frame)

        # Frame for action buttons at the bottom of the window
        button_frame = tk.Frame(self)
        button_frame.pack(side="bottom", pady=10)
        tk.Button(button_frame, text="Restore Last Settings", command=self.load_settings, font=("Arial", 10)).pack(
            side='left', padx=5)
        tk.Button(button_frame, text="Run Plotting", command=self.on_run_click, font=("Arial", 12, "bold"),
                  bg="lightblue", relief="raised", padx=10, pady=5).pack(side='left', padx=5)

    def create_main_inputs(self, frame):
        """Creates widgets for data path, dictionary file, and cycle selection."""
        labels = ["Data Directory:", "Dictionary File:", "Cycles to Plot (comma-separated):"]
        self.entries = {}
        for i, text in enumerate(labels):
            tk.Label(frame, text=text, anchor="w", font=("Arial", 10)).grid(row=i, column=0, sticky="w", pady=5)
            entry = tk.Entry(frame, width=60)
            entry.grid(row=i, column=1, padx=5, pady=5)
            self.entries[text] = entry

        # Insert default values into the entry fields
        self.entries["Data Directory:"].insert(0, self.default_data_path)
        self.entries["Dictionary File:"].insert(0, self.default_dict_file)
        self.entries["Cycles to Plot (comma-separated):"].insert(0, "25")

        # Buttons to open file dialogs for browsing
        tk.Button(frame, text="Browse", command=self.browse_data_path).grid(row=0, column=2, padx=5)
        tk.Button(frame, text="Browse", command=self.browse_dictionary_file).grid(row=1, column=2, padx=5)

        # Checkbox for handling the special case of the first cycle
        self.first_cycle_discharge_only_var = tk.BooleanVar()
        self.first_cycle_discharge_only_checkbox = tk.Checkbutton(
            frame,
            text="First cycle is discharge only",
            variable=self.first_cycle_discharge_only_var,
            onvalue=True,
            offvalue=False
        )
        self.first_cycle_discharge_only_checkbox.grid(row=3, column=0, columnspan=3, sticky="w", pady=10)
        
        # --- TAB BUTTONS: ensure controls are available on small screens ---
        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=30, column=0, columnspan=3, pady=8, sticky="w")
        tk.Button(btn_frame, text="Restore Last Settings", command=self.load_settings, font=("Arial", 10)).pack(side='left', padx=5)
        tk.Button(btn_frame, text="Run Plotting", command=self.on_run_click, font=("Arial", 10, "bold"), bg="lightblue").pack(side='left', padx=5)

    def create_plot_customization(self, frame):
        """Creates widgets for customizing the plot's appearance."""
        tk.Label(frame, text="General Plot Settings", font=("Arial", 12, "bold")).pack(fill='x', pady=(10, 5))

        # Entry for plot title
        tk.Label(frame, text="Plot Title:", anchor="w").pack(fill='x', pady=(5, 0))
        self.title_entry = tk.Entry(frame, width=80)
        self.title_entry.pack(pady=5)
        self.title_entry.insert(0, "Voltage vs. Specific Capacity")

        # Entry for x-axis label, including LaTeX for units
        tk.Label(frame, text="X-axis Label:", anchor="w").pack(fill='x', pady=(5, 0))
        self.xlabel_entry = tk.Entry(frame, width=80)
        self.xlabel_entry.pack(pady=5)
        self.xlabel_entry.insert(0, 'Specific Capacity [mAh $g^{-1}$]')

        # Entry for y-axis label, including LaTeX for units
        tk.Label(frame, text="Y-axis Label:", anchor="w").pack(fill='x', pady=(5, 0))
        self.ylabel_entry = tk.Entry(frame, width=80)
        self.ylabel_entry.pack(pady=5)
        self.ylabel_entry.insert(0, 'Voltage [V]')

        # Text area for a list of colors
        tk.Label(frame, text="Color List (comma-separated):", anchor="w").pack(fill='x', pady=(5, 0))
        self.color_text = tk.Text(frame, height=2, width=80)
        self.color_text.pack(pady=5)
        self.color_text.insert(tk.END,
                               "tab:blue, tab:orange, tab:green, tab:red, tab:purple, tab:brown, tab:pink, tab:gray, tab:olive, tab:cyan")

        # Text area for a list of alpha (transparency) values
        tk.Label(frame, text="Alpha List (comma-separated):", anchor="w").pack(fill='x', pady=(5, 0))
        self.alpha_text = tk.Text(frame, height=1, width=80)
        self.alpha_text.pack(pady=5)
        self.alpha_text.insert(tk.END, "1, 0.2, 0.8, 0.6, 0.4")

        # Entries for x-axis limits
        tk.Label(frame, text="X-axis Limits (min, max):", anchor="w").pack(fill='x', pady=(5, 0))
        xlim_frame = tk.Frame(frame)
        xlim_frame.pack(fill='x')
        self.xmin_entry = tk.Entry(xlim_frame, width=10)
        self.xmin_entry.pack(side='left', padx=5)
        tk.Label(xlim_frame, text=",").pack(side='left')
        self.xmax_entry = tk.Entry(xlim_frame, width=10)
        self.xmax_entry.pack(side='left', padx=5)

        # Entries for y-axis limits
        tk.Label(frame, text="Y-axis Limits (min, max):", anchor="w").pack(fill='x', pady=(5, 0))
        ylim_frame = tk.Frame(frame)
        ylim_frame.pack(fill='x')
        self.ymin_entry = tk.Entry(ylim_frame, width=10)
        self.ymin_entry.pack(side='left', padx=5)
        tk.Label(ylim_frame, text=",").pack(side='left')
        self.ymax_entry = tk.Entry(ylim_frame, width=10)
        self.ymax_entry.pack(side='left', padx=5)

        # --- TAB BUTTONS: ensure controls are available on small screens ---
        plot_tab_btn_frame = tk.Frame(frame)
        plot_tab_btn_frame.pack(fill='x', pady=(8, 4))
        tk.Button(plot_tab_btn_frame, text="Restore Last Settings", command=self.load_settings, font=("Arial", 10)).pack(side='left', padx=5)
        tk.Button(plot_tab_btn_frame, text="Run Plotting", command=self.on_run_click, font=("Arial", 10, "bold"), bg="lightblue").pack(side='left', padx=5)

        # Font settings
        tk.Label(frame, text="Font Settings", font=("Arial", 12, "bold")).pack(fill='x', pady=(20, 5))
        
        # Font family selection
        tk.Label(frame, text="Font Family:", anchor="w").pack(fill='x', pady=(5, 0))
        self.font_family_var = tk.StringVar()
        self.font_family_dropdown = ttk.Combobox(frame, textvariable=self.font_family_var, width=30)
        # Get available font families from matplotlib
        self.available_fonts = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])
        self.font_family_dropdown['values'] = self.available_fonts
        self.font_family_dropdown.pack(pady=5)
        # Set default font
        if 'Arial' in self.available_fonts:
            self.font_family_dropdown.set('Arial')
        else:
            self.font_family_dropdown.set(self.available_fonts[0])
        
        # Title font size
        tk.Label(frame, text="Title Font Size:", anchor="w").pack(fill='x', pady=(5, 0))
        self.title_fontsize_entry = tk.Entry(frame, width=10)
        self.title_fontsize_entry.pack(pady=5)
        self.title_fontsize_entry.insert(0, "16")

        # Axis labels font size
        tk.Label(frame, text="Axis Labels Font Size:", anchor="w").pack(fill='x', pady=(5, 0))
        self.axis_label_fontsize_entry = tk.Entry(frame, width=10)
        self.axis_label_fontsize_entry.pack(pady=5)
        self.axis_label_fontsize_entry.insert(0, "14")

        # Tick labels font size
        tk.Label(frame, text="Tick Labels Font Size:", anchor="w").pack(fill='x', pady=(5, 0))
        self.tick_label_fontsize_entry = tk.Entry(frame, width=10)
        self.tick_label_fontsize_entry.pack(pady=5)
        self.tick_label_fontsize_entry.insert(0, "12")

        # Legend font size
        tk.Label(frame, text="Legend Font Size:", anchor="w").pack(fill='x', pady=(5, 0))
        self.legend_fontsize_entry = tk.Entry(frame, width=10)
        self.legend_fontsize_entry.pack(pady=5)
        self.legend_fontsize_entry.insert(0, "10")

        # --- Save Options ---
        tk.Label(frame, text="Save Options", font=("Arial", 12, "bold")).pack(fill='x', pady=(15, 5))
        save_frame = ttk.LabelFrame(frame, text="Select which files to save", padding=(5, 5, 5, 5))
        save_frame.pack(fill='x', pady=5, padx=5)

        img_frame = tk.Frame(save_frame)
        img_frame.pack(fill='x', pady=(2, 2))
        tk.Label(img_frame, text="Images:").pack(side='left', padx=(0, 8))
        self.save_png_var = tk.BooleanVar(value=True)
        self.save_pdf_var = tk.BooleanVar(value=True)
        self.save_svg_var = tk.BooleanVar(value=True)
        tk.Checkbutton(img_frame, text="PNG", variable=self.save_png_var).pack(side='left', padx=4)
        tk.Checkbutton(img_frame, text="PDF", variable=self.save_pdf_var).pack(side='left', padx=4)
        tk.Checkbutton(img_frame, text="SVG", variable=self.save_svg_var).pack(side='left', padx=4)

        data_frame = tk.Frame(save_frame)
        data_frame.pack(fill='x', pady=(2, 2))
        tk.Label(data_frame, text="Data:").pack(side='left', padx=(0, 8))
        self.save_txt_var = tk.BooleanVar(value=True)
        self.save_netcdf_var = tk.BooleanVar(value=True)
        tk.Checkbutton(data_frame, text="TXT (flattened plot)", variable=self.save_txt_var).pack(side='left', padx=4)
        tk.Checkbutton(data_frame, text="NetCDF (per cell)", variable=self.save_netcdf_var).pack(side='left', padx=4)

        zip_frame = tk.Frame(save_frame)
        zip_frame.pack(fill='x', pady=(2, 2))
        self.save_zip_var = tk.BooleanVar(value=True)
        tk.Checkbutton(zip_frame, text="RAW ZIP (NetCDF archive)", variable=self.save_zip_var).pack(side='left', padx=4)

    def create_dict_editor(self, frame):
        """Creates widgets for loading, editing, and saving the legend dictionary file."""
        tk.Label(frame, text="Edit content of dictionary_HIPOLE.txt:", anchor="w").pack(fill='x', pady=(10, 0))
        self.dict_text = tk.Text(frame, height=20, width=80)
        self.dict_text.pack(pady=5, expand=True, fill='both')
        dict_button_frame = tk.Frame(frame)
        dict_button_frame.pack(pady=5)
        tk.Button(dict_button_frame, text="Load Dictionary", command=self.load_dictionary).pack(side='left', padx=5)
        tk.Button(dict_button_frame, text="Save Dictionary", command=self.save_dictionary).pack(side='left', padx=5)
        self.load_dictionary()

        # --- TAB BUTTONS: ensure controls are available on small screens ---
        dict_tab_btn_frame = tk.Frame(frame)
        dict_tab_btn_frame.pack(pady=5)
        tk.Button(dict_tab_btn_frame, text="Restore Last Settings", command=self.load_settings, font=("Arial", 10)).pack(side='left', padx=5)
        tk.Button(dict_tab_btn_frame, text="Run Plotting", command=self.on_run_click, font=("Arial", 10, "bold"), bg="lightblue").pack(side='left', padx=5)

    def browse_data_path(self):
        """Opens a file dialog for the user to select the data directory."""
        directory = filedialog.askdirectory(title="Select Data Directory")
        if directory:
            self.entries["Data Directory:"].delete(0, tk.END)
            self.entries["Data Directory:"].insert(0, directory)

    def browse_dictionary_file(self):
        """Opens a file dialog for the user to select the dictionary file."""
        file_path = filedialog.askopenfilename(title="Select Dictionary File", filetypes=[("Text files", "*.txt")])
        if file_path:
            self.entries["Dictionary File:"].delete(0, tk.END)
            self.entries["Dictionary File:"].insert(0, file_path)
            self.load_dictionary()

    def load_dictionary(self):
        """Loads the content of the dictionary file into the text editor."""
        dict_path = self.entries["Dictionary File:"].get()
        if not Path(dict_path).is_file():
            self.dict_text.delete(1.0, tk.END)
            self.status_label.config(text="Warning: Dictionary file not found.")
            return
        try:
            with open(dict_path, "r", encoding="cp1252", errors='ignore') as file:
                content = file.read()
                self.dict_text.delete(1.0, tk.END)
                self.dict_text.insert(tk.END, content)
                self.status_label.config(text="Dictionary loaded successfully.")
        except Exception as e:
            messagebox.showerror("Loading Error", f"Could not load dictionary: {e}")
            self.status_label.config(text="Error loading dictionary.")

    def save_dictionary(self):
        """Saves the content of the text editor to the dictionary file."""
        dict_path = self.entries["Dictionary File:"].get()
        content = self.dict_text.get(1.0, tk.END)
        try:
            # Check if the text is a valid dictionary format before saving
            ast.literal_eval(content)
            with open(dict_path, "w", encoding="cp1252", errors='ignore') as file:
                file.write(content)
            self.status_label.config(text="Dictionary saved successfully.")
        except (SyntaxError, ValueError) as e:
            messagebox.showerror("Save Error", f"Invalid dictionary format. Please check it. Error: {e}")
            self.status_label.config(text="Error: Invalid dictionary format.")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save dictionary: {e}")
            self.status_label.config(text="Error saving dictionary.")

    def save_settings(self):
        """Saves the current GUI settings to a pickle file for later use."""
        settings = {
            "data_path": self.entries["Data Directory:"].get(),
            "dictionary_name": self.entries["Dictionary File:"].get(),
            "cycle_numbers": self.entries["Cycles to Plot (comma-separated):"].get(),
            "plot_title": self.title_entry.get(),
            "xlabel_text": self.xlabel_entry.get(),
            "ylabel_text": self.ylabel_entry.get(),
            "color_list": self.color_text.get(1.0, tk.END).strip(),
            "alpha_list": self.alpha_text.get(1.0, tk.END).strip(),
            "xmin": self.xmin_entry.get(),
            "xmax": self.xmax_entry.get(),
            "ymin": self.ymin_entry.get(),
            "ymax": self.ymax_entry.get(),
            "first_cycle_discharge_only": self.first_cycle_discharge_only_var.get(),
            "title_fontsize": self.title_fontsize_entry.get(),
            "axis_label_fontsize": self.axis_label_fontsize_entry.get(),
            "tick_label_fontsize": self.tick_label_fontsize_entry.get(),
            "legend_fontsize": self.legend_fontsize_entry.get(),
            "font_family": self.font_family_var.get(),
            # Save options
            "save_png": bool(self.save_png_var.get()),
            "save_pdf": bool(self.save_pdf_var.get()),
            "save_svg": bool(self.save_svg_var.get()),
            "save_txt": bool(self.save_txt_var.get()),
            "save_netcdf": bool(self.save_netcdf_var.get()),
            "save_zip": bool(self.save_zip_var.get()),
        }
        try:
            with open(self.settings_file, 'wb') as f:
                pickle.dump(settings, f)
            self.status_label.config(text="Settings saved.")
        except Exception as e:
            self.status_label.config(text=f"Error saving settings: {e}")

    def load_settings(self):
        """Loads saved settings from the pickle file and populates the GUI fields."""
        if not Path(self.settings_file).is_file():
            self.status_label.config(text="No saved settings found.")
            return
        try:
            with open(self.settings_file, 'rb') as f:
                settings = pickle.load(f)
            # Update each GUI entry with the loaded setting
            self.entries["Data Directory:"].delete(0, tk.END)
            self.entries["Data Directory:"].insert(0, settings.get("data_path", ""))
            self.entries["Dictionary File:"].delete(0, tk.END)
            self.entries["Dictionary File:"].insert(0, settings.get("dictionary_name", ""))
            self.entries["Cycles to Plot (comma-separated):"].delete(0, tk.END)
            self.entries["Cycles to Plot (comma-separated):"].insert(0, settings.get("cycle_numbers", ""))
            self.title_entry.delete(0, tk.END)
            self.title_entry.insert(0, settings.get("plot_title", ""))
            self.xlabel_entry.delete(0, tk.END)
            self.xlabel_entry.insert(0, settings.get("xlabel_text", ""))
            self.ylabel_entry.delete(0, tk.END)
            self.ylabel_entry.insert(0, settings.get("ylabel_text", ""))
            self.color_text.delete(1.0, tk.END)
            self.color_text.insert(tk.END, settings.get("color_list", ""))
            self.alpha_text.delete(1.0, tk.END)
            self.alpha_text.insert(tk.END, settings.get("alpha_list", ""))
            self.xmin_entry.delete(0, tk.END)
            self.xmin_entry.insert(0, settings.get("xmin", ""))
            self.xmax_entry.delete(0, tk.END)
            self.xmax_entry.insert(0, settings.get("xmax", ""))
            self.ymin_entry.delete(0, tk.END)
            self.ymin_entry.insert(0, settings.get("ymin", ""))
            self.ymax_entry.delete(0, tk.END)
            self.ymax_entry.insert(0, settings.get("ymax", ""))
            self.first_cycle_discharge_only_var.set(settings.get("first_cycle_discharge_only", False))
            
            # Load font settings
            saved_font = settings.get("font_family", "Arial")
            if saved_font in self.available_fonts:
                self.font_family_dropdown.set(saved_font)
            
            # Load font sizes
            self.title_fontsize_entry.delete(0, tk.END)
            self.title_fontsize_entry.insert(0, settings.get("title_fontsize", "16"))
            self.axis_label_fontsize_entry.delete(0, tk.END)
            self.axis_label_fontsize_entry.insert(0, settings.get("axis_label_fontsize", "14"))
            self.tick_label_fontsize_entry.delete(0, tk.END)
            self.tick_label_fontsize_entry.insert(0, settings.get("tick_label_fontsize", "12"))
            self.legend_fontsize_entry.delete(0, tk.END)
            self.legend_fontsize_entry.insert(0, settings.get("legend_fontsize", "10"))

            # Load save options (defaults True)
            self.save_png_var.set(settings.get("save_png", True))
            self.save_pdf_var.set(settings.get("save_pdf", True))
            self.save_svg_var.set(settings.get("save_svg", True))
            self.save_txt_var.set(settings.get("save_txt", True))
            self.save_netcdf_var.set(settings.get("save_netcdf", True))
            self.save_zip_var.set(settings.get("save_zip", True))
            
            self.status_label.config(text="Settings loaded successfully.")
        except Exception as e:
            self.status_label.config(text=f"Error loading settings: {e}")

    def on_run_click(self):
        """
        Action handler for the "Run Plotting" button.
        It saves the current settings and calls the main plotting function with user inputs.
        """
        self.save_settings()
        # Retrieve all input values from the GUI
        data_path_str = self.entries["Data Directory:"].get()
        dictionary_name_str = self.entries["Dictionary File:"].get()
        cycle_numbers_str = self.entries["Cycles to Plot (comma-separated):"].get()
        plot_title_str = self.title_entry.get()
        xlabel_text_str = self.xlabel_entry.get()
        ylabel_text_str = self.ylabel_entry.get()
        color_list_str = self.color_text.get(1.0, tk.END).strip()
        alpha_list_str = self.alpha_text.get(1.0, tk.END).strip()
        xmin_str = self.xmin_entry.get()
        xmax_str = self.xmax_entry.get()
        ymin_str = self.ymin_entry.get()
        ymax_str = self.ymax_entry.get()
        first_cycle_discharge_only = self.first_cycle_discharge_only_var.get()
        
        # Get font sizes
        try:
            title_fontsize = int(self.title_fontsize_entry.get())
            axis_label_fontsize = int(self.axis_label_fontsize_entry.get())
            tick_label_fontsize = int(self.tick_label_fontsize_entry.get())
            legend_fontsize = int(self.legend_fontsize_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Font sizes must be valid numbers")
            return

        # Get selected font family
        font_family = self.font_family_var.get()
        
        # Set font settings in matplotlib
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': [font_family, 'DejaVu Sans', 'Arial'],
            'font.size': tick_label_fontsize,
            'axes.titlesize': title_fontsize,
            'axes.labelsize': axis_label_fontsize,
            'xtick.labelsize': tick_label_fontsize,
            'ytick.labelsize': tick_label_fontsize,
            'legend.fontsize': legend_fontsize
        })

        # Call the core plotting function with all the collected parameters
        run_plotting(
            data_path_str,
            dictionary_name_str,
            cycle_numbers_str,
            plot_title_str,
            xlabel_text_str,
            ylabel_text_str,
            color_list_str,
            alpha_list_str,
            xmin_str,
            xmax_str,
            ymin_str,
            ymax_str,
            first_cycle_discharge_only,
            close_app_on_plot_close=False,
            save_png=bool(self.save_png_var.get()),
            save_pdf=bool(self.save_pdf_var.get()),
            save_svg=bool(self.save_svg_var.get()),
            save_txt=bool(self.save_txt_var.get()),
            save_netcdf=bool(self.save_netcdf_var.get()),
            save_zip=bool(self.save_zip_var.get()),
        )  # Keep GUI open by default when plot closes


if __name__ == '__main__':
    # Create and run the GUI application
    app = VoltVsCapGUI()
    app.mainloop()
