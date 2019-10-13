import colorsys
import numpy
import numpy as np
from tkinter import *

def hsv2rgb(h,s,v):
    return tuple(int(round(i * 255)) for i in colorsys.hsv_to_rgb(h,s,v))


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def fix(val):
    return (val + 360) % 360

arousal = .2 - (.5)
valence = .6 * (-1)

x, y = cart2pol(arousal, valence)

y = fix(np.degrees(y) +270)


def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb

def change():
    #root.configure({"background": _from_rgb(tuple(np.random.choice(range(256), size=3)))})
    root.configure({"background": _from_rgb(hsv2rgb(y/360, 1, 1))})
    root.after(1000, change)

root = Tk()
change()
root.mainloop()
