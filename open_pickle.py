### authors: Philip Daubinger, Felix Nagler, Simon Feiler, Lukas Gold, Simon Stier
### v0.1, Nov 2022

import tkinter
from tkinter import filedialog
import pickle

root = tkinter.Tk()
root.title("Datenauswertung")  # Name des Fenster
file_path = filedialog.askopenfilename(initialdir=".", title="Select file", filetypes=(("pickle files", "*.pickle"), ("all files", "*.*")))  # Funktion zum Importieren des Textfiles
with open(file_path, 'rb') as file_handle:
  figx = pickle.load(file_handle)
root.destroy()
root.mainloop()

figx.show() # Show the figure, edit it, etc.!