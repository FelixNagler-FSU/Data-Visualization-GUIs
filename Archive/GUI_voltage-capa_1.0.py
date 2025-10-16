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
# This is required to show the matplotlib plot within a tkinter application.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors


# --- Refactored plotting logic into a function ---
def run_plotting(data_path_str, dictionary_name_str, cycle_numbers_str, plot_title_str, xlabel_text_str,
                 ylabel_text_str,
                 color_list_str, alpha_list_str, xmin_str, xmax_str, ymin_str, ymax_str, first_cycle_discharge_only):
    """
    This function contains the core plotting logic, refactored to accept user inputs from the GUI.
    It processes battery cycling data, calculates specific capacities, and generates a
    voltage vs. specific capacity plot.

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

    # Set default font size for the plot
    plt.rcParams.update({'font.size': 13})

    # Initialize dictionaries to store processed data and active material weights
    max_rows = 0
    all_data = {cycle: {} for cycle in cycle_numbers}
    weight_dict = {}

    # Main loop to read and process each data file
    for idx, filename in enumerate(data_file_names):
        file_path = data_path / filename

        header_line_number = None
        # Attempt to read the header and find active mass
        try:
            with open(file_path, "r", encoding="cp1252") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    # Find the header row by searching for a specific keyword
                    if "mode	ox/red	error" in line.strip():
                        header_line_number = i - 3
                    # Find and parse the "mass of active material" from the file header
                    if "mass of active material" in line.lower():
                        try:
                            parts = line.split(":")
                            number_str = parts[1].strip().split(' ')[0]
                            # Convert mass to grams and store it
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

        # Read the data file into a pandas DataFrame
        try:
            df = pd.read_table(
                filepath_or_buffer=file_path,
                sep='\t',
                header=header_line_number,
                decimal=',',
                encoding='cp1252'
            )
        except Exception as e:
            print(f"Error reading file {filename}: {e}. Skipping.")
            continue

        # Clean column names by removing leading/trailing whitespace
        df.columns = df.columns.str.strip()

        # Rename essential columns for consistency
        try:
            potential_ecell_cols = [col for col in df.columns if 'Ecell/V' in col or 'Ewe/V' in col]
            if not potential_ecell_cols:
                raise KeyError("Ecell-column not found")
            df.rename(columns={potential_ecell_cols[0]: 'Ecell'}, inplace=True)

            potential_capacity_cols = [col for col in df.columns if 'Capacity/mA.h' in col]
            if not potential_capacity_cols:
                raise KeyError("Capacity-column not found")
            df.rename(columns={potential_capacity_cols[0]: 'Capacity'}, inplace=True)

            potential_half_cycle_cols = [col for col in df.columns if 'half cycle' in col]
            if not potential_half_cycle_cols:
                raise KeyError("half cycle-column not found")
            df.rename(columns={potential_half_cycle_cols[0]: 'Half_cycle'}, inplace=True)

        except KeyError as e:
            print(f"Error finding columns in {filename}: {e}. Skipping file.")
            continue

        if df.shape[0] > max_rows:
            max_rows = df.shape[0]

        # Group data by 'Half_cycle' to separate charge and discharge phases
        grouped = df.groupby('Half_cycle')
        temp_data_dict = {}

        for half_cycle_value, group in grouped:
            if not group.empty:

                # Logic to determine charge/discharge and assign cycle number
                is_charge = False  # Default to discharge
                if first_cycle_discharge_only:
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

                # Store the data in a temporary dictionary
                temp_data_dict[volt_col] = group['Ecell'].reset_index(drop=True)
                temp_data_dict[cap_col] = group['Capacity'].reset_index(drop=True)

        # Create a DataFrame from the processed cycle data
        cycle_separated_df = pd.DataFrame(temp_data_dict)

        # Get the weight for specific capacity calculation
        weight = weight_dict.get(filename, 1)

        # Populate the main data dictionary for plotting
        for cycle in cycle_numbers:
            chCap_col = f'chCap{cycle}'
            chVolt_col = f'chVolt{cycle}'
            disCap_col = f'disCap{cycle}'
            disVolt_col = f'disVolt{cycle}'

            if chCap_col in cycle_separated_df.columns and chVolt_col in cycle_separated_df.columns:
                # Normalize capacity by active material weight
                all_data[cycle][f'{filename}_ch_cap'] = cycle_separated_df[chCap_col] / weight
                all_data[cycle][f'{filename}_ch_volt'] = cycle_separated_df[chVolt_col]

            if disCap_col in cycle_separated_df.columns and disVolt_col in cycle_separated_df.columns:
                all_data[cycle][f'{filename}_dis_cap'] = cycle_separated_df[disCap_col] / weight
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
        with open(dictionary_name, "r") as file:
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
    messagebox.showinfo("Success", "Plotting completed successfully. The graph will now be displayed.")
    plt.show()

    # Save Figures
    try:
        # Speichern der Graphen
        plot_filename_base = str(cycle_numbers_str) + str(data_path.parts[-2])
        plot_path = os.path.join(str(data_path.parent), plot_filename_base)

        plt.savefig(f'{plot_path}.png')
        plt.savefig(f'{plot_path}.svg')
        plt.savefig(f'{plot_path}.pdf')
        pickle.dump(fig, open(f'{plot_path}.pickle', 'wb'))
        print(f"Figures have been saved to {parent_dir}.")

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
        data_save_path = os.path.join(str(parent_dir),f'{plot_path}.txt')
        all_cycles_data.to_csv(data_save_path, sep='\t', index=False, float_format='%.4f')
        print(f"Plotted data saved to {data_save_path}")

    except Exception as e:
        messagebox.showerror("Error while Saving Data", f"An error occurred while saving the data to a text file: {e}")


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
            with open(dict_path, "r", encoding="utf-8") as file:
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
            with open(dict_path, "w", encoding="utf-8") as file:
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
            "first_cycle_discharge_only": self.first_cycle_discharge_only_var.get()
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

        # Call the core plotting function with all the collected parameters
        run_plotting(data_path_str, dictionary_name_str, cycle_numbers_str, plot_title_str,
                     xlabel_text_str, ylabel_text_str, color_list_str, alpha_list_str,
                     xmin_str, xmax_str, ymin_str, ymax_str, first_cycle_discharge_only)


if __name__ == '__main__':
    # Create and run the GUI application
    app = VoltVsCapGUI()
    app.mainloop()
