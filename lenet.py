from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils

import json
import os

import time
start_time = time.time()

EPOCHS = 6
BASE_BATCH_SIZE = 256
EVAL_BATCH_SIZE = BASE_BATCH_SIZE

# Declaring task_index flag which will be later used to run cluster mode
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task with in the job.")
# Declaring deploy_mode flag which will be later used to decide cluster mode
tf.app.flags.DEFINE_string("deploy_mode", "single", "either single or cluster")
FLAGS = tf.app.flags.FLAGS

single = json.dumps({
    'cluster': {
        'worker': ["10.10.1.1:2222"]
    },
    'task': {'type': 'worker', 'index': FLAGS.task_index}
})

cluster1 = json.dumps({
    'cluster': {
        'worker': ["10.10.1.1:2222", "10.10.1.2:2222"]
    },
    'task': {'type': 'worker', 'index': FLAGS.task_index}
})

cluster2 = json.dumps({
    'cluster': {
        'worker': ["10.10.1.1:2222", "10.10.1.2:2222", "10.10.1.3:2222"]
    },
    'task': {'type': 'worker', 'index': FLAGS.task_index}
})

clusterSpec = {
    "single": single,
    "cluster1": cluster1,
    "cluster2": cluster2
}

numWorker = {
    "single": 1,
    "cluster1": 2,
    "cluster2": 3
}

os.environ['TF_CONFIG'] = clusterSpec[FLAGS.deploy_mode]
NUM_WORKERS = numWorker[FLAGS.deploy_mode]
GLOBAL_BATCH_SIZE = BASE_BATCH_SIZE * NUM_WORKERS

# # Downloading the MNIST dataset
dataset = input_data.read_data_sets("MNIST_data/")

# Extracting training and testing data from the entire data set
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.mnist.load_data()

# Data needs to be reshaped to match the input layer of LeNet
# Resulting tensor will be (_, 28, 28, 1)
train_data = train_data.reshape(train_data.shape[0], 28, 28, 1)
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1)

# Values should be float so that we can get decimal points after normalization
train_data = train_data.astype('float32')
test_data = test_data.astype('float32')

# Converting grayscale [0-255] to [0-1] scale.
train_data /= 255
test_data /= 255

# Transforming training & test labels
train_labels = np_utils.to_categorical(train_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)

# Defining strategy
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

def generate_lenet_model():

    # Create a sequential model
    model = Sequential()

    # First set of CNN => RELU => POOL

    ## Layer 1 - CNN_1
    model.add(Convolution2D(
        filters=20,
        kernel_size=(5, 5),
        padding="same",
        input_shape=(28, 28, 1)))

    ## ReLU activation function
    model.add(Activation(activation="relu"))

    ## Layer 2 - Sub_Sampling_1
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Second set of CNN => RELU => POOL

    ## Layer 3 - CNN_2
    model.add(Convolution2D(
        filters=50,
        kernel_size=(5, 5),
        padding="same"))

    ## ReLU activation function
    model.add(Activation(activation="relu"))

    ## Layer 4 - Sub_Sampling_2
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Set of FC => RELU layers

    ## Flatten the network
    model.add(Flatten())

    # Fully-connected hidden layer
    model.add(Dense(500))

    ## ReLU activation function
    model.add(Activation(activation="relu"))

    # Softmax classifier

    # Fully-connected output layer
    model.add(Dense(10))

    # Softmax activation function
    model.add(Activation("softmax"))

    # Model compilation
    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(lr=0.01),
        metrics=["accuracy"])

    return model

with strategy.scope():

    # Generate model
    multi_worker_model = generate_lenet_model()

    # Train model
    multi_worker_model.fit(
        train_data,
        train_labels,
        batch_size = GLOBAL_BATCH_SIZE,
        # steps_per_epoch = 1,
        epochs = EPOCHS,
        verbose = 1)

if FLAGS.task_index == 0:
    # Evaluate model
    (loss, accuracy) = multi_worker_model.evaluate(
        test_data,
        test_labels,
        batch_size = EVAL_BATCH_SIZE,
        verbose = 3)

    # Print the model's accuracy
    print('Accuray of the model is: {}'.format(accuracy))
    print("--- %s seconds ---" % (time.time() - start_time))