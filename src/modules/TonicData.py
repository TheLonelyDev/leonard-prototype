import scipy
import numpy

tonic_emotion_matrix = {
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
    'f major': 1, #-1?
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
    'b minor': 1
}

# Create dict of numbers corresponding to pitches
pitch_dict = {0: 'C',
              1: 'C#',  # / Db',
              2: 'D',
              3: 'D#',  # / Eb',
              4: 'E',
              5: 'F',
              6: 'F#',  # / Gb',
              7: 'G',
              8: 'G#',  # / Ab',
              9: 'A',
              10: 'A#',  # / Bb',
              11: 'B'}


class TonicData():
    def translation(self, key, mode):
        return tonic_emotion_matrix['%s %s' % (key.lower(), mode.lower())]

    def ks_key(self, X):
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

        return (major.T.dot(X), minor.T.dot(X))


    def scale(self, x):
        # Get scores of every pitch throughout song from chromagram and ks_key function
        major_minor_scores = self.ks_key(x)

        major_scores = major_minor_scores[0]
        minor_scores = major_minor_scores[1]

        # Determine dominant note of key
        highest = []
        for x in range(0, len(major_scores[0])):
            i = numpy.argmax(major_scores[:, x])
            highest.append(i)

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
        the_mode = 'major' if highest.count(major_third) > highest.count(minor_third) else 'minor'

        # Get dominant note
        tonic_note = pitch_dict[tonic]

        # print(tonic_note, the_mode)
        # print(translation(tonic_note, the_mode), tonic_note, the_mode)
        return self.translation(tonic_note, the_mode)
