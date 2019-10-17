import collections
import librosa
import numpy
import numpy as np
from tkinter import *
import pyaudio
from numpy_ringbuffer import RingBuffer
import pandas
from src.modules.FeatureExtractor import FeatureExtractor
from src.modules.ModelFactory import ModelFactory
from src.modules.ColourSpace import ColourSpace
from src.modules.TonicData import TonicData

# Disable warnings
import warnings
warnings.filterwarnings("ignore")


mf = ModelFactory('C:\\Users\\Lonely\\PycharmProjects\\leonard-prototype\\src\\data', 'data4_min_16000hz_fixedmagphase.json')
extractor = FeatureExtractor()
tonicdata = TonicData()
colour = ColourSpace()

# Load a keras trained model :)
arousal_model = mf.LoadModel('arousal')
valence_model = mf.LoadModel('valence')

arousal_scaler, arousal_scalery = mf.loadScaler('arousal')
valence_scaler, valence_scalery = mf.loadScaler('valence')


sample_rate = 16000
extractor.sr = sample_rate

# Create a ringbuffer so we only keep the last 2 seconds of audio (time * sample rate) (2 * 22050)
ringBuffer = RingBuffer(int(2 * sample_rate))

# PyAudio callback
def callback(in_data, frame_count, time_info, flag):
    audio_data = numpy.fromstring(in_data, dtype=numpy.float32)

    # Resample the data from 44100 into 22050 since our dataset uses 22050
    #start = time.time()
    #audio_data = librosa.resample(audio_data, 44100, sample_rate)
    #print(time.time()-start)
    # Extend the ringbuffer
    ringBuffer.extend(audio_data)

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
stream = p.open(format=pyaudio.paFloat32, channels=1, rate=16000, input=True, output=False,
                stream_callback=callback, input_device_index=dev_index)

stream.start_stream()

def convert(x,a,b,c=0,d=1):
    """converts values in the range [a,b] to values in the range [c,d]"""
    return c + float(x-a)*float(d-c)/float(b-a)

import tensorflow
#tensorflow.compat.v1.disable_resource_variables()
tensorflow.compat.v1.disable_eager_execution()
tensorflow.compat.v1.disable_resource_variables()

print(arousal_model.metrics_names)
import time

time.sleep(2)
while True:
    extractor.y = numpy.array(ringBuffer)

    start = time.time()
    data = extractor.extract()

    # Cast this to a dataframe
    dataframe = pandas.DataFrame.from_dict(data, orient='index').T

    # Scale this data
    dataframeA = arousal_scaler.transform(numpy.array(dataframe, dtype=float))
    dataframeV = valence_scaler.transform(numpy.array(dataframe, dtype=float))

    # Predict the y values giving x
    arousal = convert(arousal_model.predict(dataframeA, batch_size=1, use_multiprocessing=True)[0][0], .15, .85, -1, 1)
    #valence = convert(valence_model.predict(dataframeV, batch_size=1, use_multiprocessing=True)[0][0], .15, .85, -1, 1)
