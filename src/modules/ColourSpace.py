import colorsys
import numpy
import scipy.constants

def convert(x,a,b,c=0,d=1):
    """converts values in the range [a,b] to values in the range [c,d]"""
    return c + float(x-a)*float(d-c)/float(b-a)

import math

def linear(a, b, t):
    return a * (1-t) + b * t

def interpolate(rgb, rgb_target, percent):
    #return linear(hue, hue_target, percent)
    r1, g1, b1 = rgb
    r2, g2, b2 = rgb_target
    p = percent
    r = int((1.0 - p) * r1 + p * r2 + 0.5)
    g = int((1.0 - p) * g1 + p * g2 + 0.5)
    b = int((1.0 - p) * b1 + p * b2 + 0.5)

    return r, g, b


class ColourSpace():
    def __init__(self):
        self.v = 1,1,1

    def hsv2rgb(self, h, s, v):
        return tuple(int(round(i * 255)) for i in colorsys.hsv_to_rgb(h, s, v))

    def cart2pol(self, x, y):
        rho = numpy.sqrt(x ** 2 + y ** 2)
        phi = numpy.arctan2(y, x)
        return (rho, phi)

    def fix(self, val):
        return (val + 360) % 360

    def from_rgb(self, rgb):
        """translates an rgb tuple of int to a tkinter friendly color code
        """
        return "#%02x%02x%02x" % rgb

    def emotion(self, valence, arousal, scale):
        x, y = self.cart2pol(((valence-.5) * -scale) * scipy.constants.golden, (arousal - .5)/scipy.constants.golden)

        y = self.fix(numpy.degrees(y) + 270) / 360
        rgb = tuple(self.hsv2rgb(y, 1, 1))
        v = tuple(interpolate(self.v, rgb, .1))
        self.v = v

        return v
