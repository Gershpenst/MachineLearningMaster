import keras
from keras import backend as K
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy

from keras.models import Model

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import cv2

class PredictionPeople:
    def __init__(self, pathModel, listClass, summary=False):
        self.model = load_model(pathModel)
        if(summary):
            self.model.summary()

        self.listClass = listClass
        self.lenClass = len(listClass)

    def predict(self, frame):
        data = []
        data.append(frame)
        dataCompress = np.array(data, dtype="float") #/ 255.0
        predictions = self.model.predict(dataCompress)

        return self.listClass[predictions.argmax(axis=1)[0]]
