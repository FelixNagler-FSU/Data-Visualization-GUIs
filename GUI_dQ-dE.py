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
    smoothing_enabled=False,
    smoothing_window=11,
    smoothing_poly=3,
    close_app_on_plot_close=False,
    # Save options
    save_png=True,
    save_pdf=True,
    save_svg=True,
    save_txt=True,
    save_netcdf=True,
    save_zip=True,
    progress_callback=None,
):
    """Core logic for dQ/dE vs E plotting.

    Behavior and key conditions/outcomes:
    - Only .mpr files are processed; others skipped.
    - Voltage column normalization: 'Ecell' -> 'Ewe' if needed.
    - If 'dQ' not present but 'Q' is, compute dQ/dE via numerical derivative using np.gradient.
    - Half-cycle grouping + first-cycle discharge-only flag controls mapping to cycle numbers.
    - Optional Savitzky-Golay smoothing applied when enabled and window/degree valid.
    - Axis limits applied only when both min and max provided.
    - Files saved into parent of data folder; filenames include experiment and data folder names.
    - NetCDF export uses unique dimension per variable to avoid size conflicts.

    Args:
        data_path_str (str): Path to the directory containing battery cycling data files.
        dictionary_name_str (str): Path to the dictionary text file for legend mapping.
        cycle_numbers_str (str): Comma-separated string of cycle numbers to plot.
        plot_title_str (str): Title for the generated plot.
        xlabel_text_str (str): Label for the x-axis.
        ylabel_text_str (str): Label for the y-axis.
        color_list_str (str): Comma-separated string of colors for the plot lines.
        alpha_list_str (str): Comma-separated string of alpha (transparency) values.
        xmin_str (str): String for the minimum x-axis limit.
        xmax_str (str): String for the maximum x-axis limit.
        ymin_str (str): String for the minimum y-axis limit.
        ymax_str (str): String for the maximum y-axis limit.
        first_cycle_discharge_only (bool): Flag to handle special first cycle logic.
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
        if progress_callback:
            progress_callback(f"Lade Datei: {filename}...")
        
        file_path = data_path / filename

        df = None
        suffix = Path(filename).suffix.lower()

        # Try .mpr via yadg first
        if suffix == '.mpr':
            try:
                import yadg
                import json
            except Exception:
                msg = f"yadg not available — falling back to header parsing for {filename}."
                print(msg)
                if progress_callback:
                    progress_callback(f"  ⚠ {msg}")
            else:
                try:
                    ds_raw = yadg.extractors.extract(filetype='eclab.mpr', path=str(file_path))
                except Exception as e:
                    msg = f"Warning: yadg failed to parse {filename}: {e}. Falling back to header parsing."
                    print(msg)
                    if progress_callback:
                        progress_callback(f"  ⚠ {msg}")
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
                            if progress_callback:
                                progress_callback(f"  → Aktivmaterialgewicht: {mg} mg ({mass_g:.4f} g)")
                    except Exception as e:
                        msg = f"Warning: error reading metadata for {filename}: {e}"
                        print(msg)
                        if progress_callback:
                            progress_callback(f"  ⚠ {msg}")

                    if mass_g is None:
                        msg = f"Warning: No active material mass found in {filename}. Skipping file."
                        print(msg)
                        if progress_callback:
                            progress_callback(f"  ⚠ {msg}")
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

                        # Find dQ, Q, Ewe (voltage) and half-cycle
                        keys = list(ds.keys()) if hasattr(ds, 'keys') else list(getattr(ds, 'data_vars', {}))
                        
                        # Look for dQ variable (differential capacity)
                        dq_var = next((k for k in keys if 'dq' in k.lower() or 'dQ' in k), None)
                        
                        # Also get Q for fallback calculation if dq is not available
                        q_var = next((k for k in keys if 'Q charge or discharge' in k or 'Q charge/discharge' in k or 'Q' == k), None)
                        
                        # Get voltage variable
                        e_var = next((k for k in keys if any(sub in k for sub in ('Ewe', 'Ecell', 'Ecell/V', 'Ewe/V'))), None)
                        
                        # Get half-cycle variable
                        half_var = next((k for k in keys if any(sub in k for sub in ('half cycle', 'Half_cycle', 'Half cycle', 'half_cycle'))), None)

                        if e_var is None or half_var is None:
                            print(f"Warning: Could not find Ewe/Half_cycle variables in {filename}. Skipping.")
                        else:
                            try:
                                # Extract arrays from the dataset
                                e_arr = np.asarray(ds[e_var].values) if hasattr(ds[e_var], 'values') else np.asarray(ds[e_var])
                                half_arr = np.asarray(ds[half_var].values) if hasattr(ds[half_var], 'values') else np.asarray(ds[half_var])
                                
                                # Try to get dQ array
                                dq_arr = None
                                q_arr = None
                                
                                if dq_var is not None:
                                    dq_arr = np.asarray(ds[dq_var].values) if hasattr(ds[dq_var], 'values') else np.asarray(ds[dq_var])
                                    df = pd.DataFrame({'dQ': dq_arr, 'Ewe': e_arr, 'Half_cycle': half_arr})
                                elif q_var is not None:
                                    # If dQ is not available, we'll calculate it from Q later
                                    q_arr = np.asarray(ds[q_var].values) if hasattr(ds[q_var], 'values') else np.asarray(ds[q_var])
                                    df = pd.DataFrame({'Q': q_arr, 'Ewe': e_arr, 'Half_cycle': half_arr})
                                else:
                                    print(f"Warning: Neither dQ nor Q found in {filename}. Skipping.")
                                    continue

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

        # Track whether the available/derived 'dQ' column is already dQ/dE (derivative)
        dq_is_dqdE = False

        # Wenn keine dQ-Spalte vorhanden ist, berechne dQ/dE mit np.gradient aus Q und E
        if 'dQ' not in df.columns and 'Q' in df.columns:
            # Nutze E (Ewe oder Ecell) und Q für die Ableitung
            e_col = 'Ewe' if 'Ewe' in df.columns else 'Ecell' if 'Ecell' in df.columns else None
            if e_col is not None:
                # np.gradient(Q, E) berechnet dQ/dE
                try:
                    df['dQ'] = np.gradient(df['Q'].astype(float).values, df[e_col].astype(float).values)
                    dq_is_dqdE = True
                except Exception:
                    # Fallback: wie bisher
                    df['dQ'] = df['Q'].diff()
                    df['dQ'].fillna(0, inplace=True)
                    dq_is_dqdE = False
        elif 'dQ' in df.columns:
            # If the dataset already provided a 'dq' variable, assume it represents dQ/dE
            # (as typical for EC-Lab exports). If this assumption is wrong for a dataset,
            # consider adjusting detection to check for explicit 'dQ/dE' variable names.
            dq_is_dqdE = True

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
                dq_col = f'chDQ{cycle_num}'
            else:
                volt_col = f'disVolt{cycle_num}'
                dq_col = f'disDQ{cycle_num}'

            # Extract voltage and dQ series
            volt_series = group.get('Ewe') if 'Ewe' in group else group.get('Ecell')
            dq_series = group.get('dQ')

            if volt_series is None or dq_series is None:
                # skip this half if necessary data missing
                continue

            # Determine dQ/dE (differential capacity)
            if dq_is_dqdE:
                # Already dQ/dE
                dQ_dE = dq_series.values
            else:
                # Compute dQ/dE from dQ by dividing by dE
                # Calculate dE (differential voltage) using numerical differentiation
                volt_array = volt_series.values
                dE_array = np.diff(volt_array)

                # To match array lengths, append last difference
                dE_array = np.append(dE_array, dE_array[-1] if len(dE_array) > 0 else 0)

                # Avoid division by zero
                with np.errstate(divide='ignore', invalid='ignore'):
                    dQ_dE = dq_series.values / dE_array
                    dQ_dE = np.where(np.isfinite(dQ_dE), dQ_dE, 0)

            # Normalize by active material weight (mass in g)
            try:
                dQ_dE_norm = dQ_dE / float(weight)
            except Exception:
                dQ_dE_norm = dQ_dE

            # Store the data
            temp_data_dict[volt_col] = volt_series.reset_index(drop=True)
            temp_data_dict[dq_col] = pd.Series(dQ_dE_norm, index=range(len(dQ_dE_norm)))

        # Create a DataFrame from the processed cycle data
        cycle_separated_df = pd.DataFrame(temp_data_dict)

        # Populate the main data dictionary for plotting
        for cycle in cycle_numbers:
            chDQ_col = f'chDQ{cycle}'
            chVolt_col = f'chVolt{cycle}'
            disDQ_col = f'disDQ{cycle}'
            disVolt_col = f'disVolt{cycle}'

            if chDQ_col in cycle_separated_df.columns and chVolt_col in cycle_separated_df.columns:
                all_data[cycle][f'{filename}_ch_dq_de'] = cycle_separated_df[chDQ_col]
                all_data[cycle][f'{filename}_ch_volt'] = cycle_separated_df[chVolt_col]

            if disDQ_col in cycle_separated_df.columns and disVolt_col in cycle_separated_df.columns:
                all_data[cycle][f'{filename}_dis_dq_de'] = cycle_separated_df[disDQ_col]
                all_data[cycle][f'{filename}_dis_volt'] = cycle_separated_df[disVolt_col]

    # Create separate DataFrames for charge and discharge data for easier plotting
    dq_de_charge_all = {cycle: pd.DataFrame({k: v for k, v in all_data[cycle].items() if '_ch_dq_de' in k}) for cycle
                           in cycle_numbers}
    voltage_charge_all = {cycle: pd.DataFrame({k: v for k, v in all_data[cycle].items() if '_ch_volt' in k}) for cycle
                          in cycle_numbers}
    dq_de_discharge_all = {cycle: pd.DataFrame({k: v for k, v in all_data[cycle].items() if '_dis_dq_de' in k}) for
                              cycle in cycle_numbers}
    voltage_discharge_all = {cycle: pd.DataFrame({k: v for k, v in all_data[cycle].items() if '_dis_volt' in k}) for
                             cycle in cycle_numbers}

    for cycle in dq_de_charge_all:
        # Replace zero values with NaN to prevent plotting them
        dq_de_charge_all[cycle].replace(0, np.nan, inplace=True)

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

    from scipy.signal import savgol_filter
    # Loop through each cycle and file to plot the data
    for i, cycle in enumerate(cycle_numbers):
        for col_idx, filename in enumerate(filtered_data_file_names):
            ch_dq_de_col = f'{filename}_ch_dq_de'
            ch_volt_col = f'{filename}_ch_volt'
            dis_dq_de_col = f'{filename}_dis_dq_de'
            dis_volt_col = f'{filename}_dis_volt'

            # Get colors and alpha values from the user-provided lists
            color_list = [c.strip() for c in color_list_str.split(',')]
            alpha_list = [float(a.strip()) for a in alpha_list_str.split(',')]
            color = color_list[col_idx % len(color_list)]
            alpha = alpha_list[i % len(alpha_list)]

            # Plot charge and discharge curves if data exists
            # --- Glättung anwenden, falls aktiviert ---
            if ch_dq_de_col in dq_de_charge_all[cycle].columns:
                x = voltage_charge_all[cycle][ch_volt_col]
                y = dq_de_charge_all[cycle][ch_dq_de_col]
                if smoothing_enabled and len(y.dropna()) >= smoothing_window and smoothing_window % 2 == 1:
                    try:
                        y_smooth = pd.Series(savgol_filter(y.fillna(0), smoothing_window, smoothing_poly))
                    except Exception:
                        y_smooth = y
                    ax1.plot(x, y_smooth, color=color, alpha=alpha)
                else:
                    ax1.plot(x, y, color=color, alpha=alpha)
            if dis_dq_de_col in dq_de_discharge_all[cycle].columns:
                x = voltage_discharge_all[cycle][dis_volt_col]
                y = dq_de_discharge_all[cycle][dis_dq_de_col]
                if smoothing_enabled and len(y.dropna()) >= smoothing_window and smoothing_window % 2 == 1:
                    try:
                        y_smooth = pd.Series(savgol_filter(y.fillna(0), smoothing_window, smoothing_poly))
                    except Exception:
                        y_smooth = y
                    ax1.plot(x, y_smooth, color=color, alpha=alpha, label=f'{legend_list[col_idx]} (Cycle {cycle})')
                else:
                    ax1.plot(x, y, color=color, alpha=alpha, label=f'{legend_list[col_idx]} (Cycle {cycle})')

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
        # Example base: dQ-dE_vs_E_[25]Experiment-data
        plot_filename_base = f"dQ-dE_vs_E_[{cycle_numbers_str}]{data_path.parts[-2]}-{data_path.parts[-1]}"
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

    # --- Save all plotted data to a text file ---
    try:
        # Create a single DataFrame to hold all the plotted data
        all_cycles_data = pd.DataFrame()
        for cycle in cycle_numbers:
            # Combine charge and discharge data for the current cycle
            cycle_charge_data = pd.DataFrame({
                f'Cycle_{cycle}_Charge_Voltage': voltage_charge_all[cycle].stack(),
                f'Cycle_{cycle}_Charge_dQ_dE': dq_de_charge_all[cycle].stack()
            }).reset_index(drop=True)

            cycle_discharge_data = pd.DataFrame({
                f'Cycle_{cycle}_Discharge_Voltage': voltage_discharge_all[cycle].stack(),
                f'Cycle_{cycle}_Discharge_dQ_dE': dq_de_discharge_all[cycle].stack()
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
                # Only ZIP desired -> use temp dir
                temp_nc_dir = os.path.join(save_dir, f"{plot_filename_base}_nc_temp")
                os.makedirs(temp_nc_dir, exist_ok=True)
            elif save_netcdf and not save_zip:
                # Save standalone NetCDF files
                permanent_nc_dir = os.path.join(save_dir, f"{plot_filename_base}_nc")
                os.makedirs(permanent_nc_dir, exist_ok=True)
            else:
                # Both True -> use temp then zip
                temp_nc_dir = os.path.join(save_dir, f"{plot_filename_base}_nc_temp")
                os.makedirs(temp_nc_dir, exist_ok=True)

        # Group files by batch using the legend_list (only if we need NetCDF)
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
                    ch_dq_de = dq_de_charge_all[cycle].get(f'{filename}_ch_dq_de')
                    ch_volt = voltage_charge_all[cycle].get(f'{filename}_ch_volt')
                    dis_dq_de = dq_de_discharge_all[cycle].get(f'{filename}_dis_dq_de')
                    dis_volt = voltage_discharge_all[cycle].get(f'{filename}_dis_volt')
                    
                    if ch_dq_de is not None and ch_volt is not None:
                        cell_data[f'cycle_{cycle}_charge_dQ_dE'] = ch_dq_de.values
                        cell_data[f'cycle_{cycle}_charge_voltage'] = ch_volt.values
                    if dis_dq_de is not None and dis_volt is not None:
                        cell_data[f'cycle_{cycle}_discharge_dQ_dE'] = dis_dq_de.values
                        cell_data[f'cycle_{cycle}_discharge_voltage'] = dis_volt.values

                if cell_data:  # Only create dataset if we have data
                    # Convert to xarray Dataset
                    # Different cycles/branches can have different lengths.
                    # Give each variable its own dimension to avoid conflicting sizes.
                    var_dict = {}
                    coords_dict = {}
                    for name, data in cell_data.items():
                        dim_name = f'point_{name}'
                        var_dict[name] = ([dim_name], data)
                        coords_dict[dim_name] = np.arange(len(data))
                    ds = xr.Dataset(var_dict, coords=coords_dict)
                    # Add metadata
                    ds.attrs['cell_number'] = cell_idx
                    ds.attrs['batch_name'] = batch_name
                    ds.attrs['filename'] = filename
                    if filename in weight_dict:
                        ds.attrs['active_material_mass_mg'] = weight_dict[filename] * 1000  # Convert back to mg
                    
                    # Create safe filename from batch name and cell number
                    safe_batch_name = "".join(c for c in batch_name if c.isalnum() or c in (' ', '-', '_')).strip()
                    nc_filename = f'{safe_batch_name}-Cell{cell_idx}.nc'
                    # Choose output directory based on config
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
            # Clean up temporary directory if used
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
class DqDeVsEGUI(tk.Tk):
    """
    Main class for the dQ/dE vs E Plotting GUI application.
    It creates the user interface, handles user interactions, and calls the plotting function.
    """

    def __init__(self):
        # Initialize the main tkinter window
        super().__init__()
        self.title("dQ/dE vs E Plotting Tool")
        self.geometry("1200x900")
        self.minsize(1000, 700)
        
        # Modern color scheme
        self.bg_color = "#f5f6fa"
        self.accent_color = "#4834df"
        self.secondary_color = "#686de0"
        self.success_color = "#26de81"
        self.text_color = "#2f3640"
        
        self.configure(bg=self.bg_color)
        
        self.settings_file = "last_dq_de_settings.pkl"
        self.default_data_path = r"C:\Users\ro45vij\Desktop\AA_Data-Processing\AA_Plotting\E4a+b+c\data"
        self.default_dict_file = "dictionary_HIPOLE.txt"
        self.create_widgets()
        # Load the last saved settings upon startup
        self.load_settings()

    def create_widgets(self):
        """Creates all the GUI widgets and organizes them using a notebook layout."""
        # Progress/Log text widget at top
        log_frame = tk.Frame(self, bg=self.bg_color)
        log_frame.pack(side="top", fill="x", padx=15, pady=(15, 5))
        tk.Label(log_frame, text="Progress Log:", font=("Segoe UI", 11, "bold"), 
                bg=self.bg_color, fg=self.text_color).pack(anchor="w", pady=(0, 5))
        self.progress_text = tk.Text(log_frame, height=8, wrap="word", state="disabled", 
                                    bg="#ffffff", fg=self.text_color, font=("Consolas", 9),
                                    relief="flat", borderwidth=2, highlightthickness=1,
                                    highlightbackground="#dfe6e9", highlightcolor=self.accent_color)
        self.progress_text.pack(fill="x", pady=(0, 0))
        
        # Status bar at the bottom of the window
        self.status_label = tk.Label(self, text="Ready", bd=0, relief="flat", anchor="w",
                                    bg="#dfe6e9", fg=self.text_color, font=("Segoe UI", 9),
                                    padx=10, pady=5)
        self.status_label.pack(side="bottom", fill="x")

        # Create a notebook widget with multiple tabs for different settings
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=self.bg_color, borderwidth=0)
        style.configure('TNotebook.Tab', padding=[20, 10], font=("Segoe UI", 10),
                       background="#ffffff", foreground=self.text_color)
        style.map('TNotebook.Tab', background=[('selected', self.accent_color)],
                 foreground=[('selected', '#ffffff')])
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=15, pady=(10, 5))

        # Tab for main input settings (data paths, cycles) - with scrollbar
        main_container = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(main_container, text="Main Settings")
        main_canvas = tk.Canvas(main_container, bg="#ffffff", highlightthickness=0)
        main_scrollbar = tk.Scrollbar(main_container, orient="vertical", command=main_canvas.yview)
        self.main_frame = tk.Frame(main_canvas, padx=20, pady=20, bg="#ffffff")
        self.main_frame.bind("<Configure>", lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all")))
        main_canvas.create_window((0, 0), window=self.main_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=main_scrollbar.set)
        main_canvas.pack(side="left", fill="both", expand=True)
        main_scrollbar.pack(side="right", fill="y")
        self.create_main_inputs(self.main_frame)

        # Tab for plot customization (title, labels, colors) - with scrollbar
        plot_container = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(plot_container, text="Plot Customization")
        plot_canvas = tk.Canvas(plot_container, bg="#ffffff", highlightthickness=0)
        plot_scrollbar = tk.Scrollbar(plot_container, orient="vertical", command=plot_canvas.yview)
        self.plot_frame = tk.Frame(plot_canvas, padx=20, pady=20, bg="#ffffff")
        self.plot_frame.bind("<Configure>", lambda e: plot_canvas.configure(scrollregion=plot_canvas.bbox("all")))
        plot_canvas.create_window((0, 0), window=self.plot_frame, anchor="nw")
        plot_canvas.configure(yscrollcommand=plot_scrollbar.set)
        plot_canvas.pack(side="left", fill="both", expand=True)
        plot_scrollbar.pack(side="right", fill="y")
        self.create_plot_customization(self.plot_frame)

        # Tab for editing the legend dictionary - with scrollbar
        dict_container = tk.Frame(self.notebook, bg="#ffffff")
        self.notebook.add(dict_container, text="Edit Dictionary")
        dict_canvas = tk.Canvas(dict_container, bg="#ffffff", highlightthickness=0)
        dict_scrollbar = tk.Scrollbar(dict_container, orient="vertical", command=dict_canvas.yview)
        self.dict_frame = tk.Frame(dict_canvas, padx=20, pady=20, bg="#ffffff")
        self.dict_frame.bind("<Configure>", lambda e: dict_canvas.configure(scrollregion=dict_canvas.bbox("all")))
        dict_canvas.create_window((0, 0), window=self.dict_frame, anchor="nw")
        dict_canvas.configure(yscrollcommand=dict_scrollbar.set)
        dict_canvas.pack(side="left", fill="both", expand=True)
        dict_scrollbar.pack(side="right", fill="y")
        self.create_dict_editor(self.dict_frame)

        # Frame for action buttons at the bottom of the window
        button_frame = tk.Frame(self, bg=self.bg_color)
        button_frame.pack(side="bottom", pady=15)
        tk.Button(button_frame, text="Restore Last Settings", command=self.load_settings,
                  font=("Segoe UI", 10), bg="#ffffff", fg=self.text_color, relief="flat",
                  padx=15, pady=8, cursor="hand2", activebackground="#dfe6e9").pack(side='left', padx=5)
        tk.Button(button_frame, text="Run Plotting", command=self.on_run_click,
                  font=("Segoe UI", 11, "bold"), bg=self.accent_color, fg="#ffffff",
                  relief="flat", padx=20, pady=10, cursor="hand2",
                  activebackground=self.secondary_color).pack(side='left', padx=5)

    def create_main_inputs(self, frame):
        """Creates widgets for data path, dictionary file, and cycle selection."""
        # Configure grid weights for responsive layout
        frame.grid_columnconfigure(1, weight=1)
        
        labels = ["Data Directory:", "Dictionary File:", "Cycles to Plot (comma-separated):"]
        self.entries = {}
        for i, text in enumerate(labels):
            tk.Label(frame, text=text, anchor="w", font=("Segoe UI", 10),
                    bg="#ffffff", fg=self.text_color).grid(row=i, column=0, sticky="w", pady=8, padx=(0, 10))
            entry = tk.Entry(frame, font=("Segoe UI", 10), relief="flat",
                           borderwidth=2, highlightthickness=1,
                           highlightbackground="#dfe6e9", highlightcolor=self.accent_color)
            entry.grid(row=i, column=1, sticky="ew", padx=(0, 5), pady=8)
            self.entries[text] = entry

        # Insert default values into the entry fields
        self.entries["Data Directory:"].insert(0, self.default_data_path)
        self.entries["Dictionary File:"].insert(0, self.default_dict_file)
        self.entries["Cycles to Plot (comma-separated):"].insert(0, "25")

        # Buttons to open file dialogs for browsing
        tk.Button(frame, text="Browse", command=self.browse_data_path,
                 font=("Segoe UI", 9), bg="#ffffff", fg=self.text_color,
                 relief="flat", padx=12, pady=6, cursor="hand2",
                 activebackground="#dfe6e9").grid(row=0, column=2, padx=5)
        tk.Button(frame, text="Browse", command=self.browse_dictionary_file,
                 font=("Segoe UI", 9), bg="#ffffff", fg=self.text_color,
                 relief="flat", padx=12, pady=6, cursor="hand2",
                 activebackground="#dfe6e9").grid(row=1, column=2, padx=5)

        # Checkbox for handling the special case of the first cycle
        self.first_cycle_discharge_only_var = tk.BooleanVar()
        self.first_cycle_discharge_only_checkbox = tk.Checkbutton(
            frame,
            text="First cycle is discharge only",
            variable=self.first_cycle_discharge_only_var,
            onvalue=True,
            offvalue=False,
            font=("Segoe UI", 9), bg="#ffffff", fg=self.text_color,
            activebackground="#ffffff", selectcolor="#ffffff"
        )
        self.first_cycle_discharge_only_checkbox.grid(row=3, column=0, columnspan=3, sticky="w", pady=10)
        
        # --- TAB BUTTONS: ensure controls are available on small screens ---
        btn_frame = tk.Frame(frame, bg="#ffffff")
        btn_frame.grid(row=30, column=0, columnspan=3, pady=8, sticky="w")
        tk.Button(btn_frame, text="Restore Last Settings", command=self.load_settings,
                 font=("Segoe UI", 9), bg="#ffffff", fg=self.text_color,
                 relief="flat", padx=12, pady=6, cursor="hand2",
                 activebackground="#dfe6e9").pack(side='left', padx=5)
        tk.Button(btn_frame, text="Run Plotting", command=self.on_run_click,
                 font=("Segoe UI", 10, "bold"), bg=self.accent_color, fg="#ffffff",
                 relief="flat", padx=15, pady=8, cursor="hand2",
                 activebackground=self.secondary_color).pack(side='left', padx=5)

    def create_plot_customization(self, frame):
        """Creates widgets for customizing the plot's appearance."""
        tk.Label(frame, text="General Plot Settings", font=("Arial", 12, "bold")).pack(fill='x', pady=(10, 5))

        # Entry for plot title
        tk.Label(frame, text="Plot Title:", anchor="w").pack(fill='x', pady=(5, 0))
        self.title_entry = tk.Entry(frame, width=80)
        self.title_entry.pack(pady=5)
        self.title_entry.insert(0, "Differential Capacity Analysis")

        # Entry for x-axis label
        tk.Label(frame, text="X-axis Label:", anchor="w").pack(fill='x', pady=(5, 0))
        self.xlabel_entry = tk.Entry(frame, width=80)
        self.xlabel_entry.pack(pady=5)
        self.xlabel_entry.insert(0, 'Voltage [V]')

        # Entry for y-axis label, including LaTeX for units
        tk.Label(frame, text="Y-axis Label:", anchor="w").pack(fill='x', pady=(5, 0))
        self.ylabel_entry = tk.Entry(frame, width=80)
        self.ylabel_entry.pack(pady=5)
        self.ylabel_entry.insert(0, 'dQ/dE [mAh $g^{-1}$ $V^{-1}$]')

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

        # --- Glättung (Savitzky-Golay) ---
        smoothing_frame = tk.LabelFrame(frame, text="Glättung (Savitzky-Golay)", padx=5, pady=5)
        smoothing_frame.pack(fill='x', pady=(15, 5))
        self.smooth_var = tk.BooleanVar()
        self.smooth_check = tk.Checkbutton(smoothing_frame, text="Glättung aktivieren", variable=self.smooth_var)
        self.smooth_check.pack(side='left', padx=5)
        tk.Label(smoothing_frame, text="Fenstergröße (ungerade):").pack(side='left', padx=5)
        self.smooth_window_entry = tk.Entry(smoothing_frame, width=5)
        self.smooth_window_entry.pack(side='left', padx=2)
        self.smooth_window_entry.insert(0, "11")
        tk.Label(smoothing_frame, text="Polynomgrad:").pack(side='left', padx=5)
        self.smooth_poly_entry = tk.Entry(smoothing_frame, width=5)
        self.smooth_poly_entry.pack(side='left', padx=2)
        self.smooth_poly_entry.insert(0, "3")

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

    def log_progress(self, message):
        """Append a timestamped message to the progress log."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.progress_text.config(state="normal")
        self.progress_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.progress_text.see(tk.END)
        self.progress_text.config(state="disabled")
        self.update_idletasks()

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
        import time
        
        # Clear previous log
        self.progress_text.config(state="normal")
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.config(state="disabled")
        
        self.log_progress("Starting plotting process...")
        self.log_progress("Reading settings from GUI...")
        
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

        # Smoothing options
        smoothing_enabled = bool(self.smooth_var.get())
        try:
            smoothing_window = int(self.smooth_window_entry.get())
            smoothing_poly = int(self.smooth_poly_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Smoothing window and polynomial degree must be integers")
            return

        # Call the core plotting function with all the collected parameters
        start_time = time.time()
        self.log_progress("Calling plotting function...")
        
        try:
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
                smoothing_enabled=smoothing_enabled,
                smoothing_window=smoothing_window,
                smoothing_poly=smoothing_poly,
                close_app_on_plot_close=False,
                save_png=bool(self.save_png_var.get()),
                save_pdf=bool(self.save_pdf_var.get()),
                save_svg=bool(self.save_svg_var.get()),
                save_txt=bool(self.save_txt_var.get()),
                save_netcdf=bool(self.save_netcdf_var.get()),
                save_zip=bool(self.save_zip_var.get()),
                progress_callback=self.log_progress,
            )  # Keep GUI open by default when plot closes
            
            elapsed_time = time.time() - start_time
            self.log_progress(f"Plotting complete! Took {elapsed_time:.1f}s")
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"ERROR: Plotting failed after {elapsed_time:.1f}s: {str(e)}"
            self.log_progress(error_msg)
            import traceback
            self.log_progress(f"  Details: {traceback.format_exc()}")


if __name__ == '__main__':
    # Create and run the GUI application
    app = DqDeVsEGUI()
    app.mainloop()
