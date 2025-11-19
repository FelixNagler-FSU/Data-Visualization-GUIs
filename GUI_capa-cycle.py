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
import sys

# Use TkAgg backend to integrate with the tkinter GUI
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

# --- Refactored plotting logic into a function ---
def run_plotting(
    data_path_str,
    dictionary_name_str,
    different_batches_str,
    number_of_cells_str,
    ce_graph_str,
    capacity_plot_mode_str,
    first_cycle_discharge_only_str,
    plot_individual_cells_str,
    color_list_str,
    marker_list_str,
    capacity_plot_title_str,
    capacity_ylabel_text_str,
    capacity_xmin_str,
    capacity_xmax_str,
    capacity_ymin_str,
    capacity_ymax_str,
    ce_plot_title_str,
    ce_ylabel_text_str,
    ce_xmin_str,
    ce_xmax_str,
    ce_ymin_str,
    ce_ymax_str,
    individual_cell_legend_suffix_str,
    # Save options
    save_png=True,
    save_pdf=True,
    save_svg=True,
    save_txt=True,
    save_netcdf=True,
    save_zip=True,
    # Fonts
    legend_font_size="12",
    legend_font_family="default",
    axis_label_font_size="13",
    axis_label_font_family="default",
    tick_label_font_size="12",
    tick_label_font_family="default",
    close_app_on_plot_close=False,
    progress_callback=None,
):
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
    
    # Store the original matplotlib font settings at the start
    original_font_family = plt.rcParams['font.family']

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

    #if len(number_of_cells) != len(data_file_names):
    if sum(number_of_cells) != len(data_file_names):
        messagebox.showwarning("Warning",
                               "Number of cells per batch does not match the number of files. Processing will continue, but this might lead to errors.")

    cell_numeration = [sum(number_of_cells[0:counter_var]) for counter_var in range(0, len(number_of_cells))]
    cell_numeration.append(sum(number_of_cells))

    max_rows = 0
    specific_charge_capacity_list = []
    specific_discharge_capacity_list = []

    # Use yadg to read .mpr files and extract per-cycle specific capacities (mAh/g)
    for counter_var in range(0, len(data_file_names)):
        file_path = data_path / data_file_names[counter_var]
        filename = data_file_names[counter_var]

        # Only process .mpr files
        if Path(filename).suffix.lower() != '.mpr':
            msg = f"Skipping non-mpr file: {filename}"
            print(msg)
            if progress_callback:
                progress_callback(f"  → {msg}")
            continue
        
        if progress_callback:
            progress_callback(f"Lade Datei: {filename}...")

        try:
            import yadg
            import json
        except Exception:
            error_msg = "yadg or json module not available. Ensure yadg is installed."
            print(error_msg)
            if progress_callback:
                progress_callback(f"ERROR: {error_msg}")
            break

        try:
            ds_raw = yadg.extractors.extract(filetype='eclab.mpr', path=str(file_path))
        except Exception as e:
            error_msg = f"Warning: failed to parse {filename} with yadg: {e}. Skipping."
            print(error_msg)
            if progress_callback:
                progress_callback(f"  ⚠ {error_msg}")
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
            error_msg = f"Warning: error reading metadata for {filename}: {e}"
            print(error_msg)
            if progress_callback:
                progress_callback(f"  ⚠ {error_msg}")

        if mass_g is None:
            error_msg = f"Warning: No active material mass found for {filename}. Skipping file."
            print(error_msg)
            if progress_callback:
                progress_callback(f"  ⚠ {error_msg}")
            continue

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
        last_charge = tmp_df[tmp_df['Q'] > 0].groupby('half', sort=False).last()['Q']
        last_discharge = tmp_df[tmp_df['Q'] < 0].groupby('half', sort=False).last()['Q'].abs()

        # Map half-cycle keys to cycles (keep first-occurrence order)
        first_cycle_discharge_only_bool = True if str(first_cycle_discharge_only).lower() in ('yes', 'true', '1') else False
        half_keys = list(dict.fromkeys(list(last_charge.index) + list(last_discharge.index)))

        charge_by_cycle = {}
        discharge_by_cycle = {}

        for h in half_keys:
            try:
                h_int = int(h)
            except Exception:
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

        
        try:
            #df_cycles['charge_mAh_g'] = (df_cycles['charge'].astype(float) / 3.6) / float(mass_g) # Convert Q (Coulomb) -> mAh and normalize by active mass (mass_g is in g)
            df_cycles['charge_mAh_g'] = (df_cycles['charge'].astype(float)) / float(mass_g)
        except Exception:
            df_cycles['charge_mAh_g'] = float('nan')
        try:
            #df_cycles['discharge_mAh_g'] = (df_cycles['discharge'].astype(float) / 3.6) / float(mass_g) # Convert Q (Coulomb) -> mAh and normalize by active mass (mass_g is in g)
            df_cycles['discharge_mAh_g'] = (df_cycles['discharge'].astype(float)) / float(mass_g) 
        except Exception:
            df_cycles['discharge_mAh_g'] = float('nan')

        # Append per-file Series (indexed 0..N-1 after reset)
        specific_charge_capacity_list.append(df_cycles['charge_mAh_g'].reset_index(drop=True))
        specific_discharge_capacity_list.append(df_cycles['discharge_mAh_g'].reset_index(drop=True))

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

    # Export data to NetCDF format for each batch (optional)
    # NetCDF is a self-describing, machine-independent data format for array-oriented scientific data
    # Include both the experiment folder (parent of "data") and the data folder name in the filename
    # Example: cap-vs-cycle_<ExperimentFolder>-<data>
    plot_name = f"cap-vs-cycle_{data_path.parts[-2]}-{data_path.parts[-1]}"
    parent_dir = Path(data_path).parent
    
    netcdf_paths = []
    # If we only want a ZIP (and not individual NetCDF files), create a temp directory
    temp_nc_dir = None
    if save_zip and not save_netcdf:
        temp_nc_dir = os.path.join(str(parent_dir), f"{plot_name}_nc_temp")
        os.makedirs(temp_nc_dir, exist_ok=True)

    if save_netcdf or save_zip:
        for counter_var in range(0, different_batches):
            try:
                # Extract batch information from the legend and file names
                batch_name = legend_list[counter_var].strip()
                batch_name_part = data_file_names[cell_numeration[counter_var]]

                # Create an xarray Dataset structure
                # This organizes the data into a self-describing dataset with:
                # - Variables: The actual data arrays (discharge/charge capacities and CE)
                # - Coordinates: The cycle numbers that index these arrays
                # - Attributes: Metadata about units and data provenance
                # All variables share the 'cycle' coordinate and have identical length by construction
                ds = xr.Dataset(
                    {
                        # Main capacity measurements with their standard deviations
                        'discharge_capacity': (['cycle'], specific_discharge_capacity[f'{batch_name_part} mean discharge capacity'].values),
                        'discharge_capacity_std': (['cycle'], specific_discharge_capacity[f'{batch_name_part} stddev discharge capacity'].values),
                        'charge_capacity': (['cycle'], specific_charge_capacity[f'{batch_name_part} mean charge capacity'].values),
                        'charge_capacity_std': (['cycle'], specific_charge_capacity[f'{batch_name_part} stddev charge capacity'].values),
                        # Coulombic efficiency data
                        'coulombic_efficiency': (['cycle'], coulombic_efficiency[f'{batch_name_part} mean'].values),
                        'coulombic_efficiency_std': (['cycle'], coulombic_efficiency[f'{batch_name_part} stddev'].values),
                    },
                    coords={
                        # Cycle numbers as coordinate variable
                        'cycle': ('cycle', np.arange(1, max_len + 1)),
                    }
                )

                # Add metadata attributes to the dataset
                # These help users understand the data's context and units
                ds.discharge_capacity.attrs['units'] = 'mAh/g'  # Specific capacity units
                ds.charge_capacity.attrs['units'] = 'mAh/g'     # Specific capacity units
                ds.coulombic_efficiency.attrs['units'] = '%'    # CE as percentage

                # Record batch information and source files
                ds.attrs['batch_name'] = batch_name  # Name from the legend dictionary
                ds.attrs['data_files'] = ', '.join(data_file_names[cell_numeration[counter_var]:cell_numeration[counter_var + 1]])  # Original data files

                # Save the dataset as a NetCDF file
                # The file name includes both the experiment name and batch identifier.
                # Sanitize the batch name to avoid illegal filename characters on Windows (e.g., '/', '\\', ':').
                safe_batch_name = "".join(c for c in batch_name if c.isalnum() or c in (' ', '-', '_')).strip()
                # Decide destination directory based on user choice
                out_dir = str(parent_dir) if save_netcdf else temp_nc_dir
                netcdf_name = os.path.join(out_dir, f'{plot_name}_{safe_batch_name}.nc')
                ds.to_netcdf(netcdf_name)
                netcdf_paths.append(netcdf_name)
                print(f"NetCDF file saved: {netcdf_name}")

            except Exception as e:
                # Log any errors but continue processing other batches
                print(f"Warning: Failed to create NetCDF for batch {batch_name}: {e}")
                continue

    # Optionally ZIP the raw NetCDF files
    if (save_netcdf or save_zip) and save_zip and netcdf_paths:
        try:
            zip_path = os.path.join(str(parent_dir), f'{plot_name}_raw_data.zip')
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for nc in netcdf_paths:
                    arcname = os.path.basename(nc)
                    zipf.write(nc, arcname)
            print(f"Raw data saved as ZIP: {zip_path}")
        except Exception as e:
            print(f"Warning: Failed to create ZIP archive: {e}")
        finally:
            # If we only created temp files (no individual NetCDF desired), clean them up
            if temp_nc_dir is not None:
                try:
                    shutil.rmtree(temp_nc_dir)
                except Exception:
                    pass

    max_cycle_dis = specific_discharge_capacity.shape[0]
    max_cycle_ce = coulombic_efficiency.shape[0]

    # --- Plotting execution ---
    # Store the original matplotlib settings
    original_font_family = plt.rcParams['font.family']
    
    # Create custom font dictionaries
    legend_font = {
        'size': legend_font_size if legend_font_size else '12',
        'family': legend_font_family if legend_font_family != "default" else None
    }
    
    axis_label_font = {
        'size': axis_label_font_size if axis_label_font_size else '13',
        'family': axis_label_font_family if axis_label_font_family != "default" else None
    }
    
    tick_label_font = {
        'size': tick_label_font_size if tick_label_font_size else '12',
        'family': tick_label_font_family if tick_label_font_family != "default" else None
    }

    # If the user requests a CE subplot (top) plus capacity subplot (bottom),
    # we build a 2-row layout; otherwise a single capacity plot is created.
    if ce_graph.lower() == 'yes':
        fig, axs = plt.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(7, 5))
        
        # Apply font settings to both axes
        for ax in axs:
            ax.tick_params(axis='both', which='major', labelsize=tick_label_font['size'])
            if tick_label_font['family']:
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_family(tick_label_font['family'])

    # Individual cell plots for Capacity
    # Condition: only if the toggle is set to 'yes'. Each cell is a thin line overlay
    # Outcome: gives per-cell traces with low alpha, while bold markers show batch means
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
    # Condition: depends on capacity_plot_mode ('discharge', 'charge', 'both')
    # Outcome: plots error bars (mean ± std) for each batch with chosen marker and color
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

    # Plots for CE (top axis) — same batching/colors as capacity
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
        axs[1].set_title(capacity_plot_title, fontsize=axis_label_font['size'], fontfamily=axis_label_font['family'])
        axs[1].set_xlabel('Cycle', fontsize=axis_label_font['size'], fontfamily=axis_label_font['family'])
        axs[1].set_ylabel(capacity_ylabel_text, fontsize=axis_label_font['size'], fontfamily=axis_label_font['family'])
        axs[1].legend(fontsize=legend_font['size'], loc=0, prop={'family': legend_font['family']})
        axs[1].grid()
        if capacity_xmin is not None and capacity_xmax is not None:
            axs[1].set_xlim(capacity_xmin, capacity_xmax)
        if capacity_ymin is not None and capacity_ymax is not None:
            axs[1].set_ylim(capacity_ymin, capacity_ymax)
        #axs[1].autoscale(enable=True, axis='both', tight=True)

        # Apply CE plot settings
        axs[0].set_title(ce_plot_title, fontsize=axis_label_font['size'], fontfamily=axis_label_font['family'])
        axs[0].set_ylabel(ce_ylabel_text, fontsize=axis_label_font['size'], fontfamily=axis_label_font['family'])
        axs[0].grid()
        axs[0].set_xticklabels([])
        if ce_xmin is not None and ce_xmax is not None:
            axs[0].set_xlim(ce_xmin, ce_xmax)
        if ce_ymin is not None and ce_ymax is not None:
            axs[0].set_ylim(ce_ymin, ce_ymax)
        #axs[0].autoscale(enable=True, axis='both', tight=True)

        plt.tight_layout()

        # Save figures and data before displaying (os._exit after show would prevent saving)
        try:
            # Build filename base that includes both the experiment folder and the data folder
            # Example: cap-vs-cycle_<Experiment>-<data>
            plot_filename_base = f"cap-vs-cycle_{data_path.parts[-2]}-{data_path.parts[-1]}"
            parent_dir = Path(data_path).parent

            # Save figures (PNG/SVG/PDF) based on user selection and a pickled figure for later editing
            plot_path = os.path.join(str(parent_dir), plot_filename_base)
            if save_png:
                plt.savefig(f"{plot_path}.png")
            if save_svg:
                plt.savefig(f"{plot_path}.svg")
            if save_pdf:
                plt.savefig(f"{plot_path}.pdf")
            pickle.dump(fig, open(os.path.join(str(parent_dir), f'{plot_filename_base}.pickle'), 'wb'))
            # Export the numeric results to CSV (TXT) if enabled
            if save_txt:
                save_file_name = os.path.join(str(parent_dir), f'{plot_filename_base}.txt')

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

        messagebox.showinfo("Success", "Plotting completed successfully. The graphs will now be displayed.")
        plt.show(block=False)
        # Don't close the figures immediately - let the user close them manually
        return

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
        ax.set_title(capacity_plot_title, fontsize=axis_label_font['size'], fontfamily=axis_label_font['family'])
        ax.set_xlabel('Cycle', fontsize=axis_label_font['size'], fontfamily=axis_label_font['family'])
        ax.set_ylabel(capacity_ylabel_text, fontsize=axis_label_font['size'], fontfamily=axis_label_font['family'])
        ax.legend(fontsize=legend_font['size'], loc=0, prop={'family': legend_font['family']})
        ax.grid()
        if capacity_xmin is not None and capacity_xmax is not None:
            ax.set_xlim(capacity_xmin, capacity_xmax)
        if capacity_ymin is not None and capacity_ymax is not None:
            ax.set_ylim(capacity_ymin, capacity_ymax)
        
        # Apply tick label font settings
            ax.tick_params(axis='both', which='major', labelsize=tick_label_font['size'])
            if tick_label_font['family']:
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_family(tick_label_font['family'])
            #ax.autoscale(enable=True, axis='both', tight=True)        plt.tight_layout()

        # Save figures and data before displaying (os._exit after show would prevent saving)
        try:
            # Build filename base that includes both the experiment folder and the data folder
            # Example: cap-vs-cycle_<Experiment>-<data>
            plot_filename_base = f"cap-vs-cycle_{data_path.parts[-2]}-{data_path.parts[-1]}"
            parent_dir = Path(data_path).parent

            # Save figures (PNG/SVG/PDF) based on user selection and a pickled figure for later editing
            plot_path = os.path.join(str(parent_dir), plot_filename_base)
            if save_png:
                plt.savefig(f"{plot_path}.png")
            if save_svg:
                plt.savefig(f"{plot_path}.svg")
            if save_pdf:
                plt.savefig(f"{plot_path}.pdf")
            pickle.dump(fig, open(os.path.join(str(parent_dir), f'{plot_filename_base}.pickle'), 'wb'))
            # Export the numeric results to CSV (TXT) if enabled
            if save_txt:
                save_file_name = os.path.join(str(parent_dir), f'{plot_filename_base}.txt')

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

        messagebox.showinfo("Success", "Plotting completed successfully. The graphs will now be displayed.")
        plt.show(block=False)
        # Don't close the figures immediately - let the user close them manually
        return

    # Saving is handled before displaying the plots so the process can exit cleanly afterwards.


# --- GUI Application Class ---
class PlottingGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advanced Plotting Tool")
        self.geometry("1200x900")
        self.minsize(1000, 700)
        
        # Modern color scheme
        self.bg_color = "#f5f6fa"
        self.accent_color = "#4834df"
        self.secondary_color = "#686de0"
        self.success_color = "#26de81"
        self.text_color = "#2f3640"
        
        self.configure(bg=self.bg_color)

        self.settings_file = "last_used_settings.pkl"
        self.default_dict_file = "dictionary_HIPOLE.txt"
        self.default_data_path = r"C:\Users\ro45vij\Desktop\AA_Data-Processing\AA_Plotting\E5_Vergleich_Swagelok-CC\data"

        self.create_widgets()
        self.load_settings()

    def create_widgets(self):
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
        progress_scrollbar = tk.Scrollbar(log_frame, command=self.progress_text.yview)
        self.progress_text.config(yscrollcommand=progress_scrollbar.set)
        
        # Status bar at bottom
        self.status_label = tk.Label(self, text="Ready", bd=0, relief="flat", anchor="w",
                                    bg="#dfe6e9", fg=self.text_color, font=("Segoe UI", 9),
                                    padx=10, pady=5)
        self.status_label.pack(side="bottom", fill="x")
        
        # Buttons at bottom
        button_frame = tk.Frame(self, bg=self.bg_color)
        button_frame.pack(side="bottom", pady=15)
        tk.Button(button_frame, text="Restore Last Settings",
                  command=self.load_settings, font=("Segoe UI", 10),
                  bg="#ffffff", fg=self.text_color, relief="flat",
                  padx=15, pady=8, cursor="hand2",
                  activebackground="#dfe6e9").pack(side='left', padx=5)
        tk.Button(button_frame, text="Run Plotting", command=self.on_run_click,
                  font=("Segoe UI", 11, "bold"), bg=self.accent_color, fg="#ffffff",
                  relief="flat", padx=20, pady=10, cursor="hand2",
                  activebackground=self.secondary_color).pack(side='left', padx=5)

        # Create a notebook (tabbed interface) with modern styling
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background=self.bg_color, borderwidth=0)
        style.configure('TNotebook.Tab', padding=[20, 10], font=("Segoe UI", 10),
                       background="#ffffff", foreground=self.text_color)
        style.map('TNotebook.Tab', background=[('selected', self.accent_color)],
                 foreground=[('selected', '#ffffff')])
        
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill='both', padx=15, pady=(10, 5))

        # Tab 1: Main Inputs - with scrollbar
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
        
        # Tab 2: Plot Customization (already has internal scrollbar)
        self.plot_frame = tk.Frame(self.notebook, padx=20, pady=20, bg="#ffffff")
        self.notebook.add(self.plot_frame, text="Plot Customization")
        self.create_plot_customization(self.plot_frame)

        # Tab 3: Dictionary Editor - with scrollbar
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

    def create_main_inputs(self, frame):
        # Configure grid weights for responsive layout
        frame.grid_columnconfigure(1, weight=1)
        
        labels = ["Data Directory:", "Dictionary File:", "Number of Batches:",
                  "Cells per Batch (comma-separated):"]
        self.entries = {}
        for i, text in enumerate(labels):
            tk.Label(frame, text=text, anchor="w", font=("Segoe UI", 10), 
                    bg="#ffffff", fg=self.text_color).grid(row=i, column=0, sticky="w", pady=8, padx=(0, 10))
            entry = tk.Entry(frame, font=("Segoe UI", 10), relief="flat", 
                           borderwidth=2, highlightthickness=1,
                           highlightbackground="#dfe6e9", highlightcolor=self.accent_color)
            entry.grid(row=i, column=1, sticky="ew", padx=(0, 5), pady=8)
            self.entries[text] = entry

        self.entries["Data Directory:"].insert(0, self.default_data_path)
        self.entries["Dictionary File:"].insert(0, self.default_dict_file)
        self.entries["Number of Batches:"].insert(0, "4")
        self.entries["Cells per Batch (comma-separated):"].insert(0,
                                                                  "1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2")

        # Browse buttons
        tk.Button(frame, text="Browse", command=self.browse_data_path,
                 font=("Segoe UI", 9), bg="#ffffff", fg=self.text_color,
                 relief="flat", padx=12, pady=6, cursor="hand2",
                 activebackground="#dfe6e9").grid(row=0, column=2, padx=5)
        tk.Button(frame, text="Browse", command=self.browse_dictionary_file,
                 font=("Segoe UI", 9), bg="#ffffff", fg=self.text_color,
                 relief="flat", padx=12, pady=6, cursor="hand2",
                 activebackground="#dfe6e9").grid(row=1, column=2, padx=5)

        # Radiobuttons for plot options
        tk.Label(frame, text="CE Graph:", anchor="w", font=("Segoe UI", 10),
                bg="#ffffff", fg=self.text_color).grid(row=4, column=0, sticky="w", pady=8, padx=(0, 10))
        self.ce_graph_var = tk.StringVar(value="Yes")
        tk.Radiobutton(frame, text="Yes", variable=self.ce_graph_var, value="Yes",
                      font=("Segoe UI", 9), bg="#ffffff", fg=self.text_color,
                      activebackground="#ffffff", selectcolor="#ffffff").grid(row=4, column=1, sticky="w")
        tk.Radiobutton(frame, text="No", variable=self.ce_graph_var, value="No",
                      font=("Segoe UI", 9), bg="#ffffff", fg=self.text_color,
                      activebackground="#ffffff", selectcolor="#ffffff").grid(row=4, column=1, padx=60, sticky="w")

        tk.Label(frame, text="Capacity Plot Mode:", anchor="w", font=("Segoe UI", 10),
                bg="#ffffff", fg=self.text_color).grid(row=5, column=0, sticky="w", pady=8, padx=(0, 10))
        self.capacity_plot_mode_var = tk.StringVar(value="both")
        tk.Radiobutton(frame, text="Discharge", variable=self.capacity_plot_mode_var, value="discharge",
                      font=("Segoe UI", 9), bg="#ffffff", fg=self.text_color,
                      activebackground="#ffffff", selectcolor="#ffffff").grid(row=5, column=1, sticky="w")
        tk.Radiobutton(frame, text="Charge", variable=self.capacity_plot_mode_var, value="charge",
                      font=("Segoe UI", 9), bg="#ffffff", fg=self.text_color,
                      activebackground="#ffffff", selectcolor="#ffffff").grid(row=5, column=1, padx=100, sticky="w")
        tk.Radiobutton(frame, text="Both", variable=self.capacity_plot_mode_var, value="both",
                      font=("Segoe UI", 9), bg="#ffffff", fg=self.text_color,
                      activebackground="#ffffff", selectcolor="#ffffff").grid(row=5, column=1, padx=180, sticky="w")

        tk.Label(frame, text="First Cycle Discharge Only:", anchor="w", font=("Segoe UI", 10),
                bg="#ffffff", fg=self.text_color).grid(row=6, column=0, sticky="w", pady=8, padx=(0, 10))
        self.first_cycle_discharge_var = tk.StringVar(value="Yes")
        tk.Radiobutton(frame, text="Yes", variable=self.first_cycle_discharge_var, value="Yes",
                      font=("Segoe UI", 9), bg="#ffffff", fg=self.text_color,
                      activebackground="#ffffff", selectcolor="#ffffff").grid(row=6, column=1, sticky="w")
        tk.Radiobutton(frame, text="No", variable=self.first_cycle_discharge_var, value="No",
                      font=("Segoe UI", 9), bg="#ffffff", fg=self.text_color,
                      activebackground="#ffffff", selectcolor="#ffffff").grid(row=6, column=1, padx=60, sticky="w")

        tk.Label(frame, text="Plot Individual Cells:", anchor="w", font=("Segoe UI", 10),
                bg="#ffffff", fg=self.text_color).grid(row=7, column=0, sticky="w", pady=8, padx=(0, 10))
        self.plot_individual_cells_var = tk.StringVar(value="No")
        tk.Radiobutton(frame, text="Yes", variable=self.plot_individual_cells_var, value="Yes",
                      font=("Segoe UI", 9), bg="#ffffff", fg=self.text_color,
                      activebackground="#ffffff", selectcolor="#ffffff").grid(row=7, column=1, sticky="w")
        tk.Radiobutton(frame, text="No", variable=self.plot_individual_cells_var, value="No",
                      font=("Segoe UI", 9), bg="#ffffff", fg=self.text_color,
                      activebackground="#ffffff", selectcolor="#ffffff").grid(row=7, column=1, padx=60, sticky="w")

    def create_plot_customization(self, frame):
        # Store default matplotlib font family for later use
        self.default_font_family = plt.rcParams['font.family']
        
        # Create a canvas with scrollbar
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Font Settings Section
        tk.Label(scrollable_frame, text="Font Settings", font=("Arial", 12, "bold")).pack(fill='x', pady=(10, 5))
        
        # Create frame for font settings
        font_frame = ttk.LabelFrame(scrollable_frame, text="Font Customization", padding=(5, 5, 5, 5))
        font_frame.pack(fill='x', pady=5, padx=5)

        # Legend Font Settings
        legend_frame = ttk.LabelFrame(font_frame, text="Legend Font", padding=(5, 5, 5, 5))
        legend_frame.pack(fill='x', pady=2)
        
        tk.Label(legend_frame, text="Font Size:").pack(side='left', padx=5)
        self.legend_font_size = tk.Entry(legend_frame, width=5)
        self.legend_font_size.pack(side='left', padx=5)
        self.legend_font_size.insert(0, "12")  # Default size
        
        tk.Label(legend_frame, text="Font Family:").pack(side='left', padx=5)
        self.legend_font_family = ttk.Combobox(legend_frame, width=15)
        self.legend_font_family['values'] = ['default'] + sorted(font_manager.get_font_names())
        self.legend_font_family.pack(side='left', padx=5)
        self.legend_font_family.set('default')

        # Axis Labels Font Settings
        axis_labels_frame = ttk.LabelFrame(font_frame, text="Axis Labels Font", padding=(5, 5, 5, 5))
        axis_labels_frame.pack(fill='x', pady=2)
        
        tk.Label(axis_labels_frame, text="Font Size:").pack(side='left', padx=5)
        self.axis_label_font_size = tk.Entry(axis_labels_frame, width=5)
        self.axis_label_font_size.pack(side='left', padx=5)
        self.axis_label_font_size.insert(0, "13")  # Default size
        
        tk.Label(axis_labels_frame, text="Font Family:").pack(side='left', padx=5)
        self.axis_label_font_family = ttk.Combobox(axis_labels_frame, width=15)
        self.axis_label_font_family['values'] = ['default'] + sorted(font_manager.get_font_names())
        self.axis_label_font_family.pack(side='left', padx=5)
        self.axis_label_font_family.set('default')

        # Tick Labels Font Settings
        tick_labels_frame = ttk.LabelFrame(font_frame, text="Tick Labels Font", padding=(5, 5, 5, 5))
        tick_labels_frame.pack(fill='x', pady=2)
        
        tk.Label(tick_labels_frame, text="Font Size:").pack(side='left', padx=5)
        self.tick_label_font_size = tk.Entry(tick_labels_frame, width=5)
        self.tick_label_font_size.pack(side='left', padx=5)
        self.tick_label_font_size.insert(0, "12")  # Default size
        
        tk.Label(tick_labels_frame, text="Font Family:").pack(side='left', padx=5)
        self.tick_label_font_family = ttk.Combobox(tick_labels_frame, width=15)
        self.tick_label_font_family['values'] = ['default'] + sorted(font_manager.get_font_names())
        self.tick_label_font_family.pack(side='left', padx=5)
        self.tick_label_font_family.set('default')

        # General Plot Settings
        tk.Label(scrollable_frame, text="General Plot Settings", font=("Arial", 12, "bold")).pack(fill='x', pady=(10, 5))

        tk.Label(scrollable_frame, text="Color List (comma-separated):", anchor="w").pack(fill='x', pady=(5, 0))
        self.color_text = tk.Text(scrollable_frame, height=4, width=80)
        self.color_text.pack(pady=5)
        self.color_text.insert(tk.END,
                               "tab:blue, tab:orange, tab:green, tab:red, tab:purple, tab:brown, tab:pink, tab:gray, tab:olive, tab:cyan")

        tk.Label(scrollable_frame, text="Marker List (comma-separated):", anchor="w").pack(fill='x', pady=(5, 0))
        self.marker_text = tk.Text(scrollable_frame, height=4, width=80)
        self.marker_text.pack(pady=5)
        self.marker_text.insert(tk.END,
                                "o, v, ^, <, >, s, p, 2, 3, 4, 8, s, p, P, *, h, H, +, x, X, D, d, |, _, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11")

        # --- Capacity Plot Settings ---
        tk.Label(scrollable_frame, text="Capacity Plot Settings", font=("Arial", 12, "bold")).pack(fill='x', pady=(15, 5))

        tk.Label(scrollable_frame, text="Capacity Plot Title:", anchor="w").pack(fill='x', pady=(5, 0))
        self.capacity_title_entry = tk.Entry(scrollable_frame, width=80)
        self.capacity_title_entry.pack(pady=5)
        self.capacity_title_entry.insert(0, "Specific Capacity and Coulombic Efficiency")

        tk.Label(scrollable_frame, text="Capacity Y-axis Label:", anchor="w").pack(fill='x', pady=(5, 0))
        self.capacity_ylabel_entry = tk.Entry(scrollable_frame, width=80)
        self.capacity_ylabel_entry.pack(pady=5)
        self.capacity_ylabel_entry.insert(0, 'Specific Capacity [mAh $g^{-1}$]')

        tk.Label(scrollable_frame, text="Individual Cell Legend Suffix:", anchor="w").pack(fill='x', pady=(5, 0))
        self.individual_cell_legend_suffix_entry = tk.Entry(scrollable_frame, width=80)
        self.individual_cell_legend_suffix_entry.pack(pady=5)
        self.individual_cell_legend_suffix_entry.insert(0, "(Cell)")

        tk.Label(scrollable_frame, text="Capacity X-axis Limits (min, max):", anchor="w").pack(fill='x', pady=(5, 0))
        cap_xlim_frame = tk.Frame(scrollable_frame)
        cap_xlim_frame.pack(fill='x')
        self.capacity_xmin_entry = tk.Entry(cap_xlim_frame, width=10)
        self.capacity_xmin_entry.pack(side='left', padx=5)
        tk.Label(cap_xlim_frame, text=",").pack(side='left')
        self.capacity_xmax_entry = tk.Entry(cap_xlim_frame, width=10)
        self.capacity_xmax_entry.pack(side='left', padx=5)

        tk.Label(scrollable_frame, text="Capacity Y-axis Limits (min, max):", anchor="w").pack(fill='x', pady=(5, 0))
        cap_ylim_frame = tk.Frame(scrollable_frame)
        cap_ylim_frame.pack(fill='x')
        self.capacity_ymin_entry = tk.Entry(cap_ylim_frame, width=10)
        self.capacity_ymin_entry.pack(side='left', padx=5)
        tk.Label(cap_ylim_frame, text=",").pack(side='left')
        self.capacity_ymax_entry = tk.Entry(cap_ylim_frame, width=10)
        self.capacity_ymax_entry.pack(side='left', padx=5)
        self.capacity_ymax_entry.insert(0, "210")

        # --- CE Plot Settings ---
        tk.Label(scrollable_frame, text="CE Plot Settings (only used if 'CE Graph' is 'Yes')", font=("Arial", 12, "bold")).pack(
            fill='x', pady=(15, 5))

        tk.Label(scrollable_frame, text="CE Plot Title:", anchor="w").pack(fill='x', pady=(5, 0))
        self.ce_title_entry = tk.Entry(scrollable_frame, width=80)
        self.ce_title_entry.pack(pady=5)
        self.ce_title_entry.insert(0, "Coulombic Efficiency")

        tk.Label(scrollable_frame, text="CE Y-axis Label:", anchor="w").pack(fill='x', pady=(5, 0))
        self.ce_ylabel_entry = tk.Entry(scrollable_frame, width=80)
        self.ce_ylabel_entry.pack(pady=5)
        self.ce_ylabel_entry.insert(0, 'CE [%]')

        tk.Label(scrollable_frame, text="CE X-axis Limits (min, max):", anchor="w").pack(fill='x', pady=(5, 0))
        ce_xlim_frame = tk.Frame(scrollable_frame)
        ce_xlim_frame.pack(fill='x')
        self.ce_xmin_entry = tk.Entry(ce_xlim_frame, width=10)
        self.ce_xmin_entry.pack(side='left', padx=5)
        tk.Label(ce_xlim_frame, text=",").pack(side='left')
        self.ce_xmax_entry = tk.Entry(ce_xlim_frame, width=10)
        self.ce_xmax_entry.pack(side='left', padx=5)

        tk.Label(scrollable_frame, text="CE Y-axis Limits (min, max):", anchor="w").pack(fill='x', pady=(5, 0))
        ce_ylim_frame = tk.Frame(scrollable_frame)
        ce_ylim_frame.pack(fill='x')
        self.ce_ymin_entry = tk.Entry(ce_ylim_frame, width=10)
        self.ce_ymin_entry.pack(side='left', padx=5)
        self.ce_ymin_entry.insert(0, "98")
        tk.Label(ce_ylim_frame, text=",").pack(side='left')
        self.ce_ymax_entry = tk.Entry(ce_ylim_frame, width=10)
        self.ce_ymax_entry.pack(side='left', padx=5)
        self.ce_ymax_entry.insert(0, "102")

        # --- Save Options ---
        tk.Label(scrollable_frame, text="Save Options", font=("Arial", 12, "bold")).pack(fill='x', pady=(15, 5))
        save_frame = ttk.LabelFrame(scrollable_frame, text="Select which files to save", padding=(5, 5, 5, 5))
        save_frame.pack(fill='x', pady=5, padx=5)

        # Image formats
        img_frame = tk.Frame(save_frame)
        img_frame.pack(fill='x', pady=(2, 2))
        tk.Label(img_frame, text="Images:").pack(side='left', padx=(0, 8))
        self.save_png_var = tk.BooleanVar(value=True)
        self.save_pdf_var = tk.BooleanVar(value=True)
        self.save_svg_var = tk.BooleanVar(value=True)
        tk.Checkbutton(img_frame, text="PNG", variable=self.save_png_var).pack(side='left', padx=4)
        tk.Checkbutton(img_frame, text="PDF", variable=self.save_pdf_var).pack(side='left', padx=4)
        tk.Checkbutton(img_frame, text="SVG", variable=self.save_svg_var).pack(side='left', padx=4)

        # Data formats
        data_frame = tk.Frame(save_frame)
        data_frame.pack(fill='x', pady=(2, 2))
        tk.Label(data_frame, text="Data:").pack(side='left', padx=(0, 8))
        self.save_txt_var = tk.BooleanVar(value=True)
        self.save_netcdf_var = tk.BooleanVar(value=True)
        tk.Checkbutton(data_frame, text="TXT (flattened plot)", variable=self.save_txt_var).pack(side='left', padx=4)
        tk.Checkbutton(data_frame, text="NetCDF (per batch)", variable=self.save_netcdf_var).pack(side='left', padx=4)

        # Raw archive
        zip_frame = tk.Frame(save_frame)
        zip_frame.pack(fill='x', pady=(2, 2))
        self.save_zip_var = tk.BooleanVar(value=True)
        tk.Checkbutton(zip_frame, text="RAW ZIP (NetCDF archive)", variable=self.save_zip_var).pack(side='left', padx=4)
        
        # Pack the canvas and scrollbar at the end
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def create_dict_editor(self, frame):
        tk.Label(frame, text="Edit content of dictionary_HIPOLE.txt:", anchor="w").pack(fill='x', pady=(10, 0))
        self.dict_text = tk.Text(frame, height=20, width=80)
        self.dict_text.pack(pady=5, expand=True, fill='both')

        dict_button_frame = tk.Frame(frame)
        dict_button_frame.pack(pady=5)
        tk.Button(dict_button_frame, text="Load Dictionary", command=self.load_dictionary).pack(side='left', padx=5)
        tk.Button(dict_button_frame, text="Save Dictionary", command=self.save_dictionary).pack(side='left', padx=5)

        self.load_dictionary()  # Load the dictionary on startup

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
            # Font settings
            "legend_font_size": self.legend_font_size.get(),
            "legend_font_family": self.legend_font_family.get(),
            "axis_label_font_size": self.axis_label_font_size.get(),
            "axis_label_font_family": self.axis_label_font_family.get(),
            "tick_label_font_size": self.tick_label_font_size.get(),
            "tick_label_font_family": self.tick_label_font_family.get(),
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
            "ce_xmax": self.ce_xmax_entry.get(),
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
        # Ensure matplotlib is using its default font family
        plt.rcParams['font.family'] = plt.rcParamsDefault['font.family']
        
        if not Path(self.settings_file).is_file():
            self.status_label.config(text="No saved settings found.")
            return

        try:
            with open(self.settings_file, 'rb') as f:
                settings = pickle.load(f)

            # Load font settings
            self.legend_font_size.delete(0, tk.END)
            self.legend_font_size.insert(0, settings.get("legend_font_size", "12"))
            self.legend_font_family.set(settings.get("legend_font_family", "default"))
            
            self.axis_label_font_size.delete(0, tk.END)
            self.axis_label_font_size.insert(0, settings.get("axis_label_font_size", "13"))
            self.axis_label_font_family.set(settings.get("axis_label_font_family", "default"))
            
            self.tick_label_font_size.delete(0, tk.END)
            self.tick_label_font_size.insert(0, settings.get("tick_label_font_size", "12"))
            self.tick_label_font_family.set(settings.get("tick_label_font_family", "default"))

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

            # Load save options (default to True for backward compatibility)
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
        import time
        import threading
        
        # Clear previous log
        self.progress_text.config(state="normal")
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.config(state="disabled")
        
        self.log_progress("Starting plotting process...")
        self.log_progress(f"Reading settings from GUI...")
        
        # Store default matplotlib settings
        original_font_family = plt.rcParams['font.family']
        
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

        # Get font settings
        legend_font_size = self.legend_font_size.get()
        legend_font_family = self.legend_font_family.get()
        axis_label_font_size = self.axis_label_font_size.get()
        axis_label_font_family = self.axis_label_font_family.get()
        tick_label_font_size = self.tick_label_font_size.get()
        tick_label_font_family = self.tick_label_font_family.get()

        start_time = time.time()
        self.log_progress("Calling plotting function...")
        
        try:
            run_plotting(data_path_str, dictionary_name_str, different_batches_str, number_of_cells_str,
                         ce_graph_str, capacity_plot_mode_str, first_cycle_discharge_only_str, plot_individual_cells_str,
                         color_list_str, marker_list_str, capacity_plot_title_str, capacity_ylabel_text_str,
                         capacity_xmin_str, capacity_xmax_str, capacity_ymin_str, capacity_ymax_str,
                         ce_plot_title_str, ce_ylabel_text_str, ce_xmin_str, ce_xmax_str, ce_ymin_str, ce_ymax_str,
                         individual_cell_legend_suffix_str,
                         # Save options
                         save_png=bool(self.save_png_var.get()),
                         save_pdf=bool(self.save_pdf_var.get()),
                         save_svg=bool(self.save_svg_var.get()),
                         save_txt=bool(self.save_txt_var.get()),
                         save_netcdf=bool(self.save_netcdf_var.get()),
                         save_zip=bool(self.save_zip_var.get()),
                         legend_font_size=legend_font_size,
                         legend_font_family=legend_font_family,
                         axis_label_font_size=axis_label_font_size,
                         axis_label_font_family=axis_label_font_family,
                         tick_label_font_size=tick_label_font_size,
                         tick_label_font_family=tick_label_font_family,
                         close_app_on_plot_close=False,
                         progress_callback=self.log_progress)
            
            elapsed_time = time.time() - start_time
            self.log_progress(f"Plotting complete! Took {elapsed_time:.1f}s")
        except Exception as e:
            elapsed_time = time.time() - start_time
            error_msg = f"ERROR: Plotting failed after {elapsed_time:.1f}s: {str(e)}"
            self.log_progress(error_msg)
            import traceback
            self.log_progress(f"  Details: {traceback.format_exc()}")



if __name__ == '__main__':
    app = PlottingGUI()
    app.mainloop()
