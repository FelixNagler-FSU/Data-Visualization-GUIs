### authors: Felix Nagler
### v0.1, Nov 2025

import sys
import tkinter
from tkinter import filedialog
import pickle

# Create a minimal Tk root only for the file dialog, keep it hidden
root = tkinter.Tk()
root.withdraw()
root.title("Datenauswertung")  # Window title

# Ask user for a pickled matplotlib figure file
file_path = filedialog.askopenfilename(
  initialdir=".",
  title="Select figure (.pickle)",
  filetypes=(("pickle files", "*.pickle"), ("all files", "*.*"))
)

# We no longer need the Tk window
try:
  root.destroy()
except Exception:
  pass

if not file_path:
  print("No file selected. Exiting.")
  sys.exit(0)

# Ensure a GUI backend is active before showing the figure
import matplotlib
try:
  matplotlib.use('TkAgg')  # use TkAgg so the figure window can display
except Exception:
  # If backend is already set, continue
  pass
import matplotlib.pyplot as plt

# Load the pickled figure and show it
with open(file_path, 'rb') as file_handle:
  figx = pickle.load(file_handle)

# Display the figure. Using plt.show(block=True) ensures the window stays open.
try:
  figx.show()
except Exception:
  # Fallback: draw via pyplot if fig.show isn't available
  try:
    plt.figure(figx.number)
  except Exception:
    pass

plt.show(block=True)