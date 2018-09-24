from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import l2 #, activity_l2
from keras.initializers import RandomUniform
import numpy
import csv
import scipy.misc
import scipy
import h5py
import numpy as np
import random as rd
import _pickle as pc
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import reuters
from keras.constraints import non_neg
from keras.preprocessing.text import Tokenizer
import models
import training_fns as tng
from keras.models import load_model
from keras import layers
# from keras import load_model_weights_hdf5

# import dataprocessing
# ~~~~~~~~~~~~~~~~~~~~~~~ MAIN FILE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():

    tng.train('MNIST','dsf',1024)


# ~~~~~~~~~~~~~~~~~~~~~~~ END MAIN FILE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~





if __name__ == "__main__":
    main()
