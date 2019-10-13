from time import sleep

import pyaudio
import librosa
import numpy as np
from numpy_ringbuffer import RingBuffer

import librosa
import numpy

import warnings
warnings.filterwarnings("ignore")

datafile = 'data3.json'
basedir = 'C:\\Users\\Lonely\\PycharmProjects\\untitled\\MEMD_audio'
import asyncio

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
    out['entropy_of_energy'] = numpy.std(entropy_of_energy)

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
    for chroma_stft in (chroma_stft):
        out[('chroma_vector%s' % counter)] = numpy.mean(chroma_stft)
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
    return out


# ring buffer will keep the last 2 seconds worth of audio
ringBuffer = RingBuffer(2 * 22050)


loop = asyncio.get_event_loop()
import matplotlib.pyplot as plt

def callback(in_data, frame_count, time_info, flag):
    audio_data = np.fromstring(in_data, dtype=np.float32)
    # we trained on audio with a sample rate of 22050 so we need to convert it
    audio_data = librosa.resample(audio_data, 44100, 22050)
    ringBuffer.extend(audio_data)

    out = load(numpy.array(ringBuffer), 22050)
    print(out['tempo'])

    return (in_data, pyaudio.paContinue)


# function that finds the index of the Soundflower
# input device and HDMI output device

p = pyaudio.PyAudio()

stream = p.open(format=pyaudio.paFloat32,
                 channels=1,
                 rate=44100,
                 input=True, output=True,
                 stream_callback=callback,
                frames_per_buffer=10240)

# start the stream
stream.start_stream()

while stream.is_active():
    sleep(200)

stream.close()
p.terminate()