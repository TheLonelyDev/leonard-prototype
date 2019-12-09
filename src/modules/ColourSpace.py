# Colour mode ideas
#   Normal
#       Normal calculation of the colour
#
#   Smooth
#       Linear interpolation
#
#   Step smooth
#       Step based interpolation
#
#
# Effect ideas (horizontal)
#   Line/strip
#       Just an uniform colour
#
#   BPM line/strip
#       Uniform colour that gets it's intensity from the BPM
#       The intensity is scaled with a logistic function with a min/max
#
#   Rolling marble
#       Calc the colour, left append this colour to the array of leds (colours will shift from left to right)
#
#   Raindrop
#       Update random array elements with the colour, decay/fade the other ones out
#
#   Waveform
#       Just a waveform spread over a line that lights leds based on the frequency % (compared to min-max)


from math import sin, cos, radians, pi, atan2, degrees
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

    def angle_to(self, p1, p2, rotation=0, clockwise=False):
        angle = degrees(atan2(p1, p2)) - rotation
        if not clockwise:
            angle = -angle
        return angle % 360


    def emotion(self, valence, arousal, scale):
        # Because a HSV hue angle is calculated counterclockwise & start at different '0 degrees' point we need to offset it
        #   Note: We need to invert valence (= * -1)

        # Use the golden ratio for faster colour switching
        x = (valence * -1)# * scipy.constants.golden_ratio
        y = (arousal) #/ scipy.constants.golden_ratio
        angle = self.angle_to(x, y, 0, False) / 360

        rgb = tuple(self.hsv2rgb(angle, 1, 1))
        v = tuple(interpolate(self.v, rgb, .1))#1/scipy.constants.golden_ratio))
        self.v = v

        return v
