import numpy
import json
import numpy as np
import csv
import os
import time

# Disable warnings, Librosa is throwing some numpy deprecation warnings and we are not interested in those
import warnings
warnings.filterwarnings("ignore")

# Data output
datafile = 'data4_min_16000hz_fixedmagphase.json'
# Data folder
basedir = 'C:\\Users\\Lonely\\PycharmProjects\\leonard-prototype\\MEMD_audio_wav16k'
# The sample length
length = 2 #2

from presets import Preset
import librosa as _librosa
librosa = Preset(_librosa)
rate = 16000
librosa['sr'] = rate

# librosa['sr'] = 11025
# librosa['n_fft'] = 1024
# librosa['hop_length'] = 256
# librosa['fmin'] = 0.5 * 11025 * 2**(-6)

# Internal kitchen past this line :O

# Calculate all audio features
def load(file, offset, duration):
    # Load the file in mono format, for a (duration) timespan and start at timeframe a (to prevent intros misbehaving + generate multiple datasets)
    y, sr = librosa.load(file, mono=True, duration=duration, offset=offset, sr=rate, res_type='kaiser_best')#, res_type='kaiser_fast')

    out = {
        'filename': file,
        'arousal': 0,
        'valence': 0,
    }

    # Compute the spectogram
    #   Avg execution time: 14ms :(
    S = librosa.stft(y)  # 3ms
    S_abs = numpy.abs(S)  # <1m
    onset_env = librosa.onset.onset_strength(S=S_abs, sr=sr)  # 6ms wuth y, 1ms with S_abs
    chroma_stft = librosa.feature.chroma_stft(S=S_abs, sr=sr)  # 6ms

    # The following features are based on https://iopscience.iop.org/article/10.1088/1757-899X/482/1/012019/pdf

    # Zero Crossing Rate
    #   Librosa: https://librosa.github.io/librosa/generated/librosa.feature.zero_crossing_rate.html
    #   Avg execution time: 2ms
    out['zero_crossing_rate'] = numpy.mean(librosa.feature.zero_crossing_rate(y=y))

    # Energy
    #   Librosa: https://librosa.github.io/librosa/generated/librosa.feature.rms.html
    #   Avg execution time: 2ms
    out['energy'] = numpy.mean(librosa.feature.rms(y=y))

    # Entropy of Energy
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.spectral_contrast.html
    #   Avg exceution time: 4ms (was 6ms)
    entropy_of_energy = librosa.feature.spectral_contrast(S=S_abs, sr=sr)
    out['entropy_of_energy'] = numpy.mean(entropy_of_energy)
    out['entropy_of_energy_std'] = numpy.std(entropy_of_energy)

    # Spectral Energy
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.spectral_bandwidth.html
    #   Avg execution time: 5ms with magphase/s_abs, 7-8ms with y
    spectral_energy = librosa.feature.spectral_bandwidth(S=S_abs, sr=sr)
    out['spectral_energy'] = numpy.mean(spectral_energy)
    out['spectral_energy_std'] = numpy.std(spectral_energy)

    # Spectral Flux
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.onset.onset_strength.html?highlight=onset_strength#librosa.onset.onset_strength
    out['spectral_flux'] = numpy.mean(onset_env)

    # Spectral Roll-off
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.spectral_rolloff.html
    out['spectral_rolloff'] = numpy.mean(librosa.feature.spectral_rolloff(S=S_abs, sr=sr))

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

    # Tempo
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.tempogram.html#librosa.feature.tempogram
    #   Librosa: http://librosa.github.io/librosa/generated/librosa.beat.plp.html#librosa.beat.plp
    #   Avg execution time: 15ms (this sounds bad, pun intended)
    out['tempo'] = numpy.mean(librosa.beat.tempo(onset_envelope=onset_env, sr=sr))

    plp = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
    print(out['tempo'])
    print((len(np.flatnonzero(librosa.util.localmax(plp)))/2)*60)
    return out

# Numpy encoder for json formatting, this is just to make sure that the data cab ve formated (numpy->python values)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Internal value for sample calc
sampleLength = length * 1000

# Internal data
tracks = []
annotations = {'arousal': {}, 'valence': {}}

def format(ms):
    return ('sample_%sms' % (ms))

def computeSample(row, start):
    # Interval
    interval = 500

    data = []
    for i in range(0, int(sampleLength/500)):
        data.append(row[format(start + (i*interval))])

    data = numpy.array(data, dtype=float)

    return numpy.mean(data)

# File loader
def annotation(file, type):
    # Open the file
    with open(file) as fh:
        # Read the csv with a , as delimiter & read as dictionary
        rd = csv.DictReader(fh, delimiter=',')

        # Loop over all entrues
        for row in rd:
            # Assign the song id to a local variable & delete it from the row
            song_id = row['song_id']
            del row['song_id']

            # Insert the data row (containing all row (headers) with sample_[x]ms data where x is the amount of ms
            annotations[type]["%s.mp3.wav" % song_id] = row

# Load in the arousal & valence annotations from the dataset
annotation('../src/data/arousal.csv', 'arousal')
annotation('../src/data/valence.csv', 'valence')

# Define how many samples per file we will calculate/fingerprint
for i in range(0, 14):# 14):
    print('Track fringerprinter range %s' % (i))

    # Loop over all tracks
    for track in os.listdir(basedir):
        # Define the start offset/time in seconds (for Librosa)
        seconds = 15 + (i * length)

        # Assign the time start (for benchmarking)
        start = time.time()

        # Load the Librosa data & arousal/valence values based on an avg/mean of the arousal/valence during this time
        track_data = load(basedir + '/' + track, seconds, length)
        track_data['arousal'] = computeSample(annotations['arousal'][track], seconds * 1000)
        track_data['valence'] = computeSample(annotations['valence'][track], seconds * 1000)

        # 'Pretty print' some output
        print('[%s]Track fingerprinter ~ @File: %s @Index: %s @Arousal: %s @Valence: %s @ElapsedPrev: %s' % (i, track, len(tracks), track_data['arousal'], track_data['valence'], time.time() - start))

        # Add the computed track_data to the tracks data
        tracks.append(track_data)

# Json encode the track data (using a NumpyEncoder just in case, this wil cast numpy data to a json formatable value)
with open(datafile, 'w') as file_out:
    json.dump(tracks, file_out, cls=NumpyEncoder)
