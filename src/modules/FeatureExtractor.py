import librosa
import numpy

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
        out = {}

        if self.file is not None:
            out = {
                'arousal': 0,
                'valence': 0,
            }

        # Compute the spectogram
        self.S = librosa.stft(self.y)
        self.S_magphase, phase = librosa.magphase(self.S)
        self.onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        self.chroma_stft = librosa.feature.chroma_stft(S=numpy.abs(self.S), sr=self.sr)

        # The following features are based on https://iopscience.iop.org/article/10.1088/1757-899X/482/1/012019/pdf

        # Zero Crossing Rate
        #   Librosa: https://librosa.github.io/librosa/generated/librosa.feature.zero_crossing_rate.html
        out['zero_crossing_rate'] = numpy.mean(librosa.feature.zero_crossing_rate(y=self.y))

        # Energy
        #   Librosa: https://librosa.github.io/librosa/generated/librosa.feature.rms.html
        out['energy'] = numpy.mean(librosa.feature.rms(y=self.y))

        # Entropy of Energy
        #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.spectral_contrast.html
        self.entropy_of_energy = librosa.feature.spectral_contrast(S=numpy.abs(self.S), sr=self.sr)
        out['entropy_of_energy'] = numpy.mean(self.entropy_of_energy)
        out['entropy_of_energy_std'] = numpy.std(self.entropy_of_energy)

        # Spectral Energy
        #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.spectral_bandwidth.html
        self.spectral_energy = librosa.feature.spectral_bandwidth(S=self.S_magphase)
        out['spectral_energy'] = numpy.mean(self.spectral_energy)
        out['spectral_energy_std'] = numpy.std(self.spectral_energy)

        # Spectral Flux
        #   Librosa: http://librosa.github.io/librosa/generated/librosa.onset.onset_strength.html?highlight=onset_strength#librosa.onset.onset_strength
        out['spectral_flux'] = numpy.mean(self.onset_env)

        # Spectral Roll-off
        #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.spectral_rolloff.html
        out['spectral_rolloff'] = numpy.mean(librosa.feature.spectral_rolloff(S=self.S_magphase, sr=self.sr))

        # MFCC
        #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.mfcc.html
        counter = 1
        for mfcc in (librosa.feature.mfcc(y=self.y, sr=self.sr, n_mels=13)):
            out[('mfcc%s' % counter)] = numpy.mean(mfcc)
            counter = counter + 1

        # Chroma Vector (Pitch vector)
        #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.chroma_stft.html
        counter = 1
        for chroma in (self.chroma_stft):
            out[('chroma_vector%s' % counter)] = numpy.mean(chroma)
            counter = counter + 1

        # Chroma Deviation (Pitch deviation)
        #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.chroma_stft.html
        out['chroma_deviation'] = numpy.std(self.chroma_stft)

        # Tempo
        #   Librosa: http://librosa.github.io/librosa/generated/librosa.feature.tempogram.html#librosa.feature.tempogram
        #   Librosa: http://librosa.github.io/librosa/generated/librosa.beat.plp.html#librosa.beat.plp
        out['tempo'] = numpy.mean(librosa.beat.tempo(onset_envelope=self.onset_env, sr=self.sr))

        return out
