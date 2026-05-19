import tkinter as tk
from tkinter import filedialog as fd


def askopenfilename_hidden_root(title,
                                initialdir,
                                filetypes=None):
    root = tk.Tk()
    root.withdraw()
    try:
        dialog_kwargs = {
            "title": title,
            "initialdir": str(initialdir),
        }
        if filetypes is not None:
            dialog_kwargs["filetypes"] = filetypes
        return fd.askopenfilename(**dialog_kwargs)
    finally:
        root.destroy()