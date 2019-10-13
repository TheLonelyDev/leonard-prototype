import collections
from time import sleep, time
import librosa
import tensorflow
import time

from src.modules.FeatureExtractor import FeatureExtractor
from src.modules.ModelFactory import ModelFactory

# Disable warnings
import warnings

import pyaudio
from numpy_ringbuffer import RingBuffer

warnings.filterwarnings("ignore")

# Imports
import pandas




# Create & train the models; save them for later

tra = {
    'c major': 1,
    'c minor': -1,

    'c# minor': -1,
    'c# major': -1,
    'db minor': -1,
    'db major': -1,

    'd major': 1,
    'd minor': -1,


    'd# minor': -1,
    'd# major': 1,
    'eb minor': -1,
    'eb major': 1,


    'e major': 1,
    'e minor': -1,
    'f major': -1,
    'f minor': -1,


    'f# major': 1,
    'f# minor': -1,
    'gb major': 1,
    'gb minor': -1,

    'g major': 1,
    'g minor': -1,


    'g# major': -1,
    'g# minor': -1,
    'ab major': -1,
    'ab minor': -1,


    'a major': 1,
    'a minor': 1,

    'a# major': 1,
    'a# minor': -1,
    'bb major': 1,
    'bb minor': -1,

    'b major': -1,
    'b minor': -1
}

import string
def translation(key, mode):
    return tra['%s %s' % (key.lower(), mode.lower())]


import scipy
def ks_key(X):
    """
    Function that estimates the key from a
    pitch class distribution
    :param X: pitch-class energy distribution:
    np.ndarray
    :return: 2 arrays of correlation scores for
    major and minor keys: np.ndarray (shape=(12,)),
    np.ndarray (shape=(12,))
    """
    X = scipy.stats.zscore(X)

    # Coefficients from Kumhansl and Schmuckler
    # as reported here: http://rnhart.net/articles/key-finding/
    major = numpy.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    major = scipy.stats.zscore(major)

    minor = numpy.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    minor = scipy.stats.zscore(minor)

    # Generate all rotations of major
    major = scipy.linalg.circulant(major)
    minor = scipy.linalg.circulant(minor)

    return(major.T.dot(X), minor.T.dot(X))

def scale(x):
    # Get scores of every pitch throughout song from chromagram and ks_key function
    major_minor_scores = ks_key(x)

    major_scores = major_minor_scores[0]
    minor_scores = major_minor_scores[1]

    # Determine dominant note of key
    highest = []
    for x in range(0, len(major_scores[0])):
        i = numpy.argmax(major_scores[:, x])
        highest.append(i)

    # Create dict of numbers corresponding to pitches
    pitch_dict = {0: 'C',
                  1: 'C#',# / Db',
                  2: 'D',
                  3: 'D#',# / Eb',
                  4: 'E',
                  5: 'F',
                  6: 'F#',# / Gb',
                  7: 'G',
                  8: 'G#',# / Ab',
                  9: 'A',
                  10: 'A#', # / Bb',
                  11: 'B'}

    # Get mode (major or minor)
    highest_count = 0
    top_num = 0

    for x in range(0, 12):
        curr_count = highest.count(x)
        if curr_count > highest_count:
            highest_count = curr_count
            top_num = x

    # Get number representing base note
    tonic = top_num

    # Major third follows pattern of 2 whole steps
    major_third = tonic + 4

    # Minor third follows pattern of whole step, 1/2 step
    minor_third = tonic + 3

    # Find which third (major or minor) appears more
    if highest.count(major_third) > highest.count(minor_third):
        the_mode = 'major'
        mode_int = 1
    else:
        the_mode = 'minor'
        mode_int = 2

    # Get dominant note
    tonic_note = pitch_dict[tonic]

    #print(tonic_note, the_mode)
    #print(translation(tonic_note, the_mode), tonic_note, the_mode)
    return translation(tonic_note, the_mode)



# Create a ringbuffer so we only keep the last 2 seconds of audio (time * sample rate) (2 * 22050)
ringBuffer = RingBuffer(2 * 22050)

# Create a deque with a maxlen of 1
# With this deque we can communicate between the PyAudio callback & the main threa
q = collections.deque(maxlen=1)

import numpy
# PyAudio callback
def callback(in_data, frame_count, time_info, flag):
    audio_data = numpy.fromstring(in_data, dtype=numpy.float32)

    # Resample the data from 44100 into 22050 since our dataset uses 22050
    audio_data = librosa.resample(audio_data, 44100, 22050)

    # Extend the ringbuffer
    ringBuffer.extend(audio_data)

    # Insert the data in the ringbuffer (this will later be accessed in the main thread)
    q.clear()
    q.append(ringBuffer)

    return (in_data, pyaudio.paContinue)


print('Creating PyAudio instance')
print('\t\tPlease note: we are using the windows stereo mix in this setup')
p = pyaudio.PyAudio()

# Determine the stereo mix audio driver
dev_index = 2
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if (dev['name'] == 'Stereo Mix (Realtek(R) Audio' and dev['hostApi'] == 0):
        dev_index = dev['index']

# Create & start PyAudio stream
#   Note: this is a mono channel
#   Adjust frames_per_buffer to adjust accuracy
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=44100, input=True, output=False,
                stream_callback=callback, input_device_index=dev_index)

stream.start_stream()

mf = ModelFactory('C:\\Users\\Lonely\\PycharmProjects\\leonard-prototype\\src\\data', 'data4_min.json')
extractor = FeatureExtractor()

# Load a keras trained model :)
arousal_model = mf.CreateArousalModel()#mf.LoadModel('arousal')
valence_model = mf.CreateValenceModel()#mf.LoadModel('valence')

arousal_scaler, arousal_scalery = mf.loadScaler('arousal')
valence_scaler, valence_scalery = mf.loadScaler('valence')

def convert(x,a,b,c=0,d=1):
    """converts values in the range [a,b] to values in the range [c,d]"""
    return c + float(x-a)*float(d-c)/float(b-a)

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

def _from_rgb(rgb):
    """translates an rgb tuple of int to a tkinter friendly color code
    """
    return "#%02x%02x%02x" % rgb

def change():
    #root.configure({"background": _from_rgb(tuple(np.random.choice(range(256), size=3)))})
    if len(numpy.array(ringBuffer)) is not 0:
        start = time.time()
        extractor.y(numpy.array(ringBuffer))
        extractor.sr(22050)

        data = extractor.extract()
        # Cast this to a dataframe
        dataframe = pandas.DataFrame.from_dict(data, orient='index').T

        # Scale this data
        dataframeA = arousal_scaler.transform(numpy.array(dataframe, dtype=float))
        dataframeV = valence_scaler.transform(numpy.array(dataframe, dtype=float))

        # Predict the y values giving x
        arousal = convert(arousal_model.predict(dataframeA)[0][0], .15, .85, 0, 1)
        valence = convert(valence_model.predict(dataframeV)[0][0], .15, .85, 0, 1)

        root.winfo_toplevel().title('Arousal/energy: %s Key power: %s Scale: %s' % (round(arousal, 2), round(valence, 2), scale))

        scale = 1

        x, y = cart2pol((valence - (.5)) * (scale), arousal - (.5))

        y = fix(np.degrees(y) + 270)

        root.configure({"background": _from_rgb(hsv2rgb(y/360, 1, 1))})

    root.after(10, change)

root = Tk()
change()
root.mainloop()
