try:
    import os
    import tkinter as tk
    import tkinter.ttk as ttk
    from tkinter import filedialog
except ImportError:
    import Tkinter as tk
    import ttk
    import tkFileDialog as filedialog

def img_path():
    root = tk.Tk()
    filez = filedialog.askopenfilenames(parent=root,title='Choose a file')
    return (root.tk.splitlist(filez))

