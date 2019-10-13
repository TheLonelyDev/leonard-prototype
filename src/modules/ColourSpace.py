import colorsys
import numpy

def convert(x,a,b,c=0,d=1):
    """converts values in the range [a,b] to values in the range [c,d]"""
    return c + float(x-a)*float(d-c)/float(b-a)

class ColourSpace():
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
        x, y = self.cart2pol((valence - .5) * scale, arousal - .5)

        y = self.fix(numpy.degrees(y) + 270)
        print(arousal, convert(arousal, 0, 1, 0.7, 1))
        return self.hsv2rgb(y / 360, 1, convert(arousal, 0, 1, 0.5, 1))
