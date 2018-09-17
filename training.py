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
from keras.models import load_model
from keras import layers
# from keras import load_model_weights_hdf5

# import dataprocessing
# ~~~~~~~~~~~~~~~~~~~~~~~ MAIN FILE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():

    train('MNIST','ffn',32)


def save_weights(load_name):

    model = load_model(load_name)
    i=1
    for layer in model.layers:
        if (layer.get_config()["name"].startswith("dense")):
            weights = layer.get_weights() # list of numpy arrays
            print('weights dimensions: ',len(weights[0]),'x',len(weights[0][0]))
            np.savetxt("weights_%s.csv"%str(i),weights[0], delimiter=",")
            i=i+1

def summary(load_name):
    model = load_model(load_name)
    print(model.summary())
    for layer in model.layers:
        if (layer.get_config()["name"].startswith("dense")):
            print(layer.get_config()['name'],':',layer.get_config()['units'])


def train(dataset, model_type, hidden_neurons1, hidden_neurons2=None):
    depth = 1
    epochs = 10
    batch_size = 32
    val_split = 0.2

    if hidden_neurons2!=None:
        layers = 2
    else:
        layers = 1

    # TRAIN, TEST or SAVE
    # mode = 'SAVE'
    # dataset = 'REUT'
    # load_name = 'Model.ffn512.01-0.5500.hdf5'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~ DEAL WITH THE DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # if mode == 'TRAIN':
    if dataset=='MNIST':
        width = height = 28
        depth = 1
        classes = 10
        (X_train, y_train), (X_test, y_test) = mnist.load_data()


            # val_row = int(X_train.shape[0] * 0.8)

    if dataset == 'REUT':
        width =5000
        depth = height = 1
        classes = 46
        max_words = 5000
        (X_train, y_train), (X_test, y_test) = reuters.load_data(path="reuters.npz",
                                                     num_words=None,
                                                     skip_top=0,
                                                     maxlen=max_words,
                                                     test_split=0.2,
                                                     seed=113,
                                                     start_char=1,
                                                     oov_char=2,
                                                     index_from=3)

        print('Shape: ',len(X_train[0]))

    if dataset == 'MATH':
        width = height = 32
        depth = 1
        classes = 369
        X_train = pc.load(open("/vol/bitbucket/mrs516/hasy_data/x_train.pkl","rb"))
        X_test = pc.load(open("/vol/bitbucket/mrs516/hasy_data/x_test.pkl","rb"))
        y_train = np.genfromtxt('/vol/bitbucket/mrs516/hasy_data/y_train.csv')
        y_test = np.genfromtxt('/vol/bitbucket/mrs516/hasy_data/y_test.csv')


    if dataset == 'REUT':
        tok = Tokenizer(num_words=max_words)
        X_train = tok.sequences_to_matrix(X_train, mode='binary')
        X_test = tok.sequences_to_matrix(X_test, mode='binary')

    X_train = X_train.reshape(X_train.shape[0], width, height, depth).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], width, height, depth).astype('float32')

    print ('shape: ',X_train.shape[0],'x',X_train.shape[1])


        # Rescale the inputs to [0,1]
    if dataset != 'REUT':
        X_test /=255
        X_train /=255

    y_train = np_utils.to_categorical(y_train, classes)
    y_test = np_utils.to_categorical(y_test, classes)

    # Selecting the preprocessing if activated
    generate = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        rotation_range=0.0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    if model_type=='dsf':
        our_model = models.dsf(dataset,hidden_neurons1,hidden_neurons2)
    elif model_type=='ffn':
        our_model = models.ffn(dataset,hidden_neurons1,hidden_neurons2)
    else:
        our_model = models.cnn(dataset)


    files='Model.'+model_type+'.{epoch:02d}-{val_acc:.4f}.hdf5'
    ckpt = keras.callbacks.ModelCheckpoint(files, monitor = 'val_loss',verbose=1, save_best_only=True, mode='auto')
    stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
    our_model.fit_generator(generate.flow(X_train, y_train), steps_per_epoch = X_train.shape[0], epochs=epochs,
                            validation_data = (X_test,y_test), callbacks = [ckpt,stop])

    # test run on the test classes
    # if mode == 'TEST':
    #     model = keras.models.load_model(load_name)
    #     incorrect_classes = np.nonzero(model.predict(X_test, batch_size=None, verbose=0, steps=None) != y_test)
    #     print(incorrect_classes)
    #
    # if mode == 'SAVE':
    #     model = load_model(load_name)
    #     i=1
    #     for layer in model.layers:
    #         if (layer.get_config()["name"].startswith("dense")):
    #             weights = layer.get_weights() # list of numpy arrays
    #             print('weights dimensions: ',len(weights[0]),'x',len(weights[0][0]))
    #             np.savetxt("weights_mathffn512_%s.csv"%str(i),weights[0], delimiter=",")
    #             i=i+1
    #
    # if mode == 'SUMMARY':
    #     model = load_model(load_name)
    #     print(model.summary())
    #     for layer in model.layers:
    #         if (layer.get_config()["name"].startswith("dense")):
    #             print(layer.get_config()['name'],':',layer.get_config()['units'])

if __name__ == "__main__":
    main()
