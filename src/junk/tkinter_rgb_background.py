from tkinter import *
import numpy as np

def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb

def change():
    root.configure({"background": _from_rgb(tuple(np.random.choice(range(256), size=3)))})
    root.after(1000, change)

root = Tk()
change()
root.mainloop()
