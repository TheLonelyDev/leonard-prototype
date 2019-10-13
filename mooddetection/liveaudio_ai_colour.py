import collections
from time import sleep, time
from sklearn.externals import joblib
import librosa
import numpy
import os
import keras
import tensorflow as tf
import tensorflow
import time

# Disable warnings
import warnings

import pyaudio
from numpy_ringbuffer import RingBuffer

warnings.filterwarnings("ignore")

# Datafile
datafile = './data4.json'

# Imports
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import json
import numpy
import pandas
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

scalerX = MinMaxScaler()
scalerY = MinMaxScaler()

def loadScaler(type):
    # Create scalers (load the from saved scaler files if they exist)
    return (joblib.load('%s.scalerX' % type) if os.path.isfile('%s.scalerX' % type) else MinMaxScaler(),
    joblib.load('%s.scalerY' % type) if os.path.isfile('%s.scalerY' % type) else MinMaxScaler())

def saveScaler(type):
    # Save the scales
    joblib.dump(scalerX, '%s.scalerX' % type)
    joblib.dump(scalerY, '%s.scalerY' % type)

def LoadDatafile():
    # Load the json file into a pandas dataframe
    # Drop the filename column
    # Shuffle the data for good luck
    #   NOTE: this is not scientifically proven!
    return shuffle(pandas.read_json(datafile).drop(['filename'], axis=1))


def StandardScaling(type, x, y):
    # Scale the datasets & convert to numpy arrays
    x = scalerX.fit_transform(numpy.array(x, dtype=float))
    y = scalerY.fit_transform(numpy.array(y, dtype=float))

    saveScaler(type)

    # Split into train & test datasets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    return x_train, x_test, y_train, y_test


# Load data & return arousal & valence (y) (2,) shape
def StandardScaledAV():
    data = LoadDatafile()

    return StandardScaling('av', data.iloc[:, 2:], data.iloc[:, :2])


# Load data & only return arousal (y) (1,) shape
def StandardScaledArousal():
    data = LoadDatafile()

    return StandardScaling('arousal', data.iloc[:, 2:], data.iloc[:, :1])


# Load data & only return valence (y) (1,) shape
def StandardScaledValence():
    data = LoadDatafile()

    return StandardScaling('valence', data.iloc[:, 2:], data.iloc[:, 1:2])


# Create a keras MES model
def MeanSquared(x, opt):
    tf.keras.backend.clear_session()

    model = models.Sequential([
        #  Create a relu activation layer
        layers.Dense(39, activation='relu', input_shape=(x.shape[1],), kernel_initializer='normal'),

        # Add a dropout of 25% to prevent overfitting
        layers.Dropout(.25),

        # Create 3 relu based layers
        layers.Dense(19, activation='relu'),
        layers.Dense(9, activation='relu'),
        layers.Dense(4, activation='relu'),

        # Create the output based on tanh, this makes the output less likely to be around the centre point
        layers.Dense(1, activation='tanh')
    ])

    model.compile(optimizer="adam", loss='mean_squared_logarithmic_error', metrics=['mse', 'mae'])

    return model


# Arousal value refitting, only for plotting purposes
def RefitArousalY(y):
    return numpy.insert(numpy.array(y), 0, 0, axis=1)


# Valence value refitting, only for plotting purposes
def RefitValenceY(y):
    return numpy.insert(numpy.array(y), 1, 0, axis=1)


# Create a arousal based model
def CreateArousalModel():
    x_train, x_test, y_train, y_test = StandardScaledArousal()

    model = MeanSquared(x_train, 'adam')
    history = model.fit(x_train, y_train, epochs=200, verbose=True, validation_data=(x_test, y_test))

    model.save('./arousal_model.h5')

    return model


# Create a valence based model
def CreateValenceModel():
    x_train, x_test, y_train, y_test = StandardScaledValence()

    model = MeanSquared(x_train, 'adam')
    history = model.fit(x_train, y_train, epochs=200, verbose=True, validation_data=(x_test, y_test))

    model.save('./valence_model.h5')

    return model


# Create & train the models; save them for later

#CreateValenceModel()

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

# Calculate all audio features
def load(y, sr):
    out = {
    }
    # Compute the spectogram
    S = librosa.stft(y)
    S_magphase, phase = librosa.magphase(S)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    chroma_stft = librosa.feature.chroma_stft(S=numpy.abs(S), sr=sr)

    # The following features are based on https://iopscience.iop.org/article/10.1088/1757-899X/482/1/012019/pdf

    # Zero Crossing Rate
    #   Librosa: https://librosa.github.io/librosa/generated/librosa.feature.zero_crossing_rate.html
    out['zero_crossing_rate'] = numpy.mean(librosa.feature.zero_crossing_rate(y=y))

    # Energy
    #   Librosa: https://librosa.github.io/librosa/generated/librosa.feature.rms.html
    out['energy'] = numpy.mean(librosa.feature.rms(y=y))

    # Entropy of Energy
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.spectral_contrast.html
    entropy_of_energy = librosa.feature.spectral_contrast(S=numpy.abs(S), sr=sr)
    out['entropy_of_energy'] = numpy.mean(entropy_of_energy)
    out['entropy_of_energy_std'] = numpy.std(entropy_of_energy)

    # Spectral Energy
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.spectral_bandwidth.html
    spectral_energy = librosa.feature.spectral_bandwidth(S=S_magphase)
    out['spectral_energy'] = numpy.mean(spectral_energy)
    out['spectral_energy_std'] = numpy.std(spectral_energy)

    # Spectral Flux
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.onset.onset_strength.html?highlight=onset_strength#librosa.onset.onset_strength
    out['spectral_flux'] = numpy.mean(onset_env)

    # Spectral Roll-off
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.spectral_rolloff.html
    out['spectral_rolloff'] = numpy.mean(librosa.feature.spectral_rolloff(S=S_magphase, sr=sr))

    # MFCC
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
    counter = 1
    for mfcc in (librosa.feature.mfcc(y=y, sr=sr, n_mels=13)):
        out[('mfcc%s' % counter)] = numpy.mean(mfcc)
        counter = counter + 1

    # Chroma Vector (Pitch vector)
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.chroma_stft.html
    counter = 1
    for chroma in (chroma_stft):
        out[('chroma_vector%s' % counter)] = numpy.mean(chroma)
        counter = counter + 1

    # Chroma Deviation (Pitch deviation)
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.chroma_stft.html
    out['chroma_deviation'] = numpy.std(chroma_stft)

    # Tonnetz
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.tonnetz.html
    out['tonnetz'] = numpy.mean(librosa.feature.tonnetz(y=y, sr=sr))

    # Own feautures:
    #   - tempo
    #   - harmonic pitch
    #   - percusive pitch

    # Tempo & Tempo deviation
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.tempogram.html#librosa.feature.tempogram
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.beat.plp.html#librosa.beat.plp
    out['tempo'] = numpy.mean(librosa.beat.tempo(onset_envelope=onset_env, sr=sr))

    # Harmonic & percusive pitch
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.chroma_stft.html
    #   Librosa: https://librosa.github.io/librosa/generated/librosa.effects.hpss.html
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    out['harmonic_pitch'] = numpy.mean(librosa.feature.chroma_stft(y=y_harmonic, sr=sr))
    out['percusive_pitch'] = numpy.mean(librosa.feature.chroma_stft(y=y_percussive, sr=sr))

#    print("harmonic")
    #scale(librosa.feature.chroma_cqt(y=y_harmonic, sr=sr))

    out['scale'] = scale(chroma_stft)


    return out


# Create a ringbuffer so we only keep the last 2 seconds of audio (time * sample rate) (2 * 22050)
ringBuffer = RingBuffer(2 * 22050)

# Create a deque with a maxlen of 1
# With this deque we can communicate between the PyAudio callback & the main threa
q = collections.deque(maxlen=1)


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

# Load a keras trained model :)
arousal_model = tensorflow.keras.models.load_model('./arousal_model.h5')#CreateArousalModel()#tensorflow.keras.models.load_model('./arousal_model.h5')
valence_model = tensorflow.keras.models.load_model('./valence_model.h5')#CreateValenceModel()

arousal_scaler, arousal_scalery = loadScaler('arousal')
valence_scaler, valence_scalery = loadScaler('valence')

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
        data = load(numpy.array(ringBuffer), 22050)
        scale = data['scale']
        del data['scale']
        # Cast this to a dataframe
        dataframe = pandas.DataFrame.from_dict(data, orient='index').T

        # Scale this data
        dataframeA = arousal_scaler.transform(numpy.array(dataframe, dtype=float))
        dataframeV = valence_scaler.transform(numpy.array(dataframe, dtype=float))

        # Predict the y values giving x
        arousal = convert(arousal_model.predict(dataframeA)[0][0], .15, .85, 0, 1)
        valence = convert(valence_model.predict(dataframeV)[0][0], .15, .85, 0, 1)

       # print('Arousal/energy: %s\nKey power: %s' % (round(arousal, 2), round(valence, 2)))
        root.winfo_toplevel().title('Arousal/energy: %s Key power: %s Scale: %s' % (round(arousal, 2), round(valence, 2), scale))

        x, y = cart2pol((valence - (.5)) * (scale), arousal - (.5))

        y = fix(np.degrees(y) + 270)

        root.configure({"background": _from_rgb(hsv2rgb(y/360, 1, 1))})

    root.after(10, change)

root = Tk()
change()
root.mainloop()

# run = True
# try:
#     while run:
#         # Get the features from Librosa at a 22050 sample rate
#         if len(numpy.array(ringBuffer)) is not 0:
#             data = load(numpy.array(ringBuffer), 22050)
#
#             # Cast this to a dataframe
#             dataframe = pandas.DataFrame.from_dict(data, orient='index').T
#
#             # Scale this data
#             dataframeA = arousal_scaler.transform(numpy.array(dataframe, dtype=float))
#             dataframeV = valence_scaler.transform(numpy.array(dataframe, dtype=float))
#
#             # Predict the y values giving x
#             arousal = convert(arousal_model.predict(dataframeA)[0][0], .15, .85, 0, 1)
#             valence = convert(valence_model.predict(dataframeV)[0][0], .15, .85, 0, 1)
#             print('Arousal/energy: %s\nKey power: %s' % (round(arousal, 2), round(valence, 2)))
# except (KeyboardInterrupt, SystemExit):
#     stream.stop_stream()
#     stream.close()
#     run = False
#
# stream.close()
# p.terminate()

