from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import pandas
import numpy
import tensorflow
import os
from tensorflow.keras import models
from tensorflow.keras import layers

import wandb
from wandb.keras import WandbCallback
wandb.init(project="leonard")

epoch = 500
#tensorflow.compat.v1.disable_resource_variables()
tensorflow.compat.v1.disable_eager_execution()

class ModelFactory():
    def __init__(self, datadir, datafile):
        self.datadir = datadir
        self.datafile = datafile

    def pathFormat(self, type, extension):
        return '%s/%s.%s' % (self.datadir, type, extension)

    def loadScaler(self, type):
        # Create scalers (load the from saved scaler files if they exist)
        return (joblib.load(self.pathFormat(type, 'scalerX')) if os.path.isfile(self.pathFormat(type, 'scalerX')) else MinMaxScaler(),
                joblib.load(self.pathFormat(type, 'scalerY')) if os.path.isfile(self.pathFormat(type, 'scalerY')) else MinMaxScaler())

    def saveScaler(self, type):
        # Save the scales
        joblib.dump(self.scalerX, self.pathFormat(type, 'scalerX'))
        joblib.dump(self.scalerY, self.pathFormat(type, 'scalerY'))

    def MinMaxScaling(self, type, x, y):
        # Reset the scalers
        self.scalerX = MinMaxScaler()
        self.scalerY = MinMaxScaler()

        # Scale the datasets & convert to numpy arrays
        x = self.scalerX.fit_transform(numpy.array(x, dtype=float))
        y = self.scalerY.fit_transform(numpy.array(y, dtype=float))

        self.saveScaler(type)

        # Split into train & test datasets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

        return x_train, x_test, y_train, y_test

    def LoadDatafile(self):
        # Load the json file into a pandas dataframe
        # Drop the filename column
        # Shuffle the data for good luck
        #   NOTE: this is not scientifically proven!
        return shuffle(pandas.read_json('%s/%s' % (self.datadir, self.datafile)).drop(['filename'], axis=1))

    # Load data & return arousal & valence (y) (2,) shape
    def MinMaxScaledAV(self):
        data = self.LoadDatafile()

        return self.MinMaxScaling('av', data.iloc[:, 2:], data.iloc[:, :2])

    # Load data & only return arousal (y) (1,) shape
    def MinMaxScaledArousal(self):
        data = self.LoadDatafile()
        return self.MinMaxScaling('arousal', data.iloc[:, 2:], data.iloc[:, :1])

    # Load data & only return valence (y) (1,) shape
    def MinMaxScaledValence(self):
        data = self.LoadDatafile()

        return self.MinMaxScaling('valence', data.iloc[:, 2:], data.iloc[:, 1:2])

    # Create a keras MES model
    def MeanSquared(self, x, opt):
        tensorflow.keras.backend.clear_session()

        model = models.Sequential([
            layers.Dense(32, activation='relu', input_shape=(x.shape[1],), kernel_initializer='normal'),
            layers.Dense(16, activation='relu'),
            layers.Dense(8, activation='relu'),
            layers.Dense(1, activation='tanh')
        ])

        model.compile(optimizer="adam", loss='mean_squared_logarithmic_error', metrics=['mse', 'mae', 'msle', tensorflow.keras.metrics.Precision(), tensorflow.keras.metrics.Recall()])

        return model

    # Arousal value refitting, only for plotting purposes
    def RefitArousalY(self, y):
        return numpy.insert(numpy.array(y), 0, 0, axis=1)

    # Valence value refitting, only for plotting purposes
    def RefitValenceY(self, y):
        return numpy.insert(numpy.array(y), 1, 0, axis=1)

    # Create a arousal based model
    def CreateArousalModel(self):
        x_train, x_test, y_train, y_test = self.MinMaxScaledArousal()
        x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2)

        model = self.MeanSquared(x_train, 'adam')
        history = model.fit(x_train, y_train, epochs=epoch, verbose=True, validation_data=(x_validate, y_validate), callbacks=[WandbCallback()])
        print(model.summary())
        print(model.evaluate(x_test, y_test))

        model.save(self.pathFormat('arousal', '.h5'))

        return model

    # Create a valence based model
    def CreateValenceModel(self):
        x_train, x_test, y_train, y_test = self.MinMaxScaledValence()
        x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.2)

        model = self.MeanSquared(x_train, 'adam')
        history = model.fit(x_train, y_train, epochs=epoch, verbose=True, validation_data=(x_validate, y_validate), callbacks=[WandbCallback()])
        print(model.summary())
        print(model.evaluate(x_test, y_test))

        model.save(self.pathFormat('valence', '.h5'))

        return model

    def LoadModel(self, type):
        return tensorflow.keras.models.load_model(self.pathFormat(type, '.h5'))