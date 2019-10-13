import colorsys
import numpy

class ColourSpace():
    def hsv2rgb(self, h, s, v):
        return tuple(int(round(i * 255)) for i in colorsys.hsv_to_rgb(h, s, v))

    def cart2pol(self, x, y):
        rho = numpy.sqrt(x ** 2 + y ** 2)
        phi = numpy.arctan2(y, x)
        return (rho, phi)

    def fix(self, val):
        return (val + 360) % 360

    def _from_rgb(self, rgb):
        """translates an rgb tuple of int to a tkinter friendly color code
        """
        return "#%02x%02x%02x" % rgb