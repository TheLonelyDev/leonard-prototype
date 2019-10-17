import librosa
import numpy
import time

# from presets import Preset
# import librosa as _librosa
# librosa = Preset(_librosa)
# librosa['sr'] = 11025
# librosa['n_fft'] = 1024
# librosa['hop_length'] = 256
# librosa['fmin'] = 0.5 * 11025 * 2**(-6)

class FeatureExtractor():
    def __init__(self):
        self.file = None
        self.sr = 22050
        self.y = None

    def loadFile(self, file, duration, offset):
        self.file = file

        # Load the file in mono format, for a (duration) timespan and start at timeframe a (to prevent intros misbehaving + generate multiple datasets)
        self.y, self.sr = librosa.load(file, mono=True, duration=duration, offset=offset, sr=self.sr, res_type='kaiser_best')

    def y(self, y):
        self.y = y

    def extract(self):
        #   Avg execution time: 50ms :(
        #       Use s_abs instead of magphase, reduces load by 7-10ms!
        #       Use s_abs instead of y for onset_strength, reduces load by 6ms
        out = {}

        if self.file is not None:
            out = {
                'arousal': 0,
                'valence': 0,
            }
        y = self.y
        sr = self.sr

        # Compute the spectogram
        #   Avg execution time: 14ms :(
        S = librosa.stft(y) #3ms
        S_abs = numpy.abs(S) #<1m
        onset_env = librosa.onset.onset_strength(S=S_abs, sr=sr) #6ms wuth y, 1ms with S_abs
        chroma_stft = librosa.feature.chroma_stft(S=S_abs, sr=sr) #6ms

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

        return out
