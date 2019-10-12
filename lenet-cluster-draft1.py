from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np

import json
import os

BUFFER_SIZE = 10000
BATCH_SIZE = 64

# Scaling MNIST data from (0, 255] to (0., 1.]
def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255
  return image, label

datasets, info = tfds.load(name='mnist',
                           with_info=True,
                           as_supervised=True)

train_datasets_unbatched = datasets['train'].map(scale).cache().shuffle(BUFFER_SIZE)
train_datasets = train_datasets_unbatched.batch(BATCH_SIZE)

def build_and_compile_cnn_model():
    # Create a sequential model
    model = Sequential()

    # Add the first convolution layer
    model.add(Convolution2D(
        filters=20,
        kernel_size=(5, 5),
        padding="same",
        input_shape=(28, 28, 1)))

    # Add a ReLU activation function
    model.add(Activation(
        activation="relu"))

    # Add a pooling layer
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)))

    # Add the second convolution layer
    model.add(Convolution2D(
        filters=50,
        kernel_size=(5, 5),
        padding="same"))

    # Add a ReLU activation function
    model.add(Activation(
        activation="relu"))

    # Add a second pooling layer
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)))

    # Flatten the network
    model.add(Flatten())

    # Add a fully-connected hidden layer
    model.add(Dense(500))

    # Add a ReLU activation function
    model.add(Activation(
        activation="relu"))

    # Add a fully-connected output layer
    model.add(Dense(10))

    # Add a softmax activation function
    model.add(Activation("softmax"))
    # Compile the network

    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(lr=0.01),
        metrics=["accuracy"])

    return model

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

NUM_WORKERS = 2
# Here the batch size scales up by number of workers since
# `tf.data.Dataset.batch` expects the global batch size. Previously we used 64,
# and now this becomes 128.
GLOBAL_BATCH_SIZE = 64 * NUM_WORKERS
train_datasets = train_datasets_unbatched.batch(GLOBAL_BATCH_SIZE)
with strategy.scope():
  multi_worker_model = build_and_compile_cnn_model()
# Train the model
multi_worker_model.fit(
    train_datasets,
    train_labels,
    batch_size = 128,
    nb_epoch = 20,
	  verbose = 1)