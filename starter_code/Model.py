### YOUR CODE HERE

# import torch
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from ImageUtils import parse_record
from Network3 import MyNetwork

"""This script defines the training, validation and testing process.
"""


class MyModel(object):
    def __init__(self, configs):
        self.configs = configs
        self.network = MyNetwork(configs)
        self.model = None

    def train(self, x_train, y_train, training_config, x_valid=None, y_valid=None):
        num_epochs = training_config["num_epochs"]
        batch_size = training_config["batch_size"]
        verbose = training_config["verbose"]
        learning_rate = training_config["learning_rate"]
        momentum = training_config["momentum"]
        num_steps = int(x_train.shape[0] / batch_size)
        x_train = np.apply_along_axis(
            func1d=parse_record, arr=x_train, axis=1, training=True
        )
        x_valid = np.apply_along_axis(
            func1d=parse_record, arr=x_valid, axis=1, training=True
        )
        optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.model = self.network()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        datagen = ImageDataGenerator(
            width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
        )
        history = self.model.fit_generator(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=num_steps,
            epochs=num_epochs,
            validation_data=(x_valid, y_valid),
            verbose=verbose,
        )
        saved_model_folder = os.path.abspath(self.configs["save_dir"])
        self.model.save(saved_model_folder, overwrite=True, save_format="tf")
        return history

    def evaluate(self, x, y):
        x = np.apply_along_axis(func1d=parse_record, arr=x, axis=1, training=True)
        saved_model_folder = os.path.abspath(self.configs["save_dir"])
        self.model = tf.keras.models.load_model(saved_model_folder)
        loss, accc = self.model.evaluate(x, y)
        print(
            f"validation done : the loss and the accuracy reported by the model were {loss} and {accc}"
        )

    def predict_prob(self, x):
        saved_model_folder = os.path.abspath(self.configs["save_dir"])
        if os.path.exists(saved_model_folder):
            self.model = tf.keras.models.load_model(saved_model_folder)
        x = np.apply_along_axis(func1d=parse_record, arr=x, axis=1, training=False)
        return self.model.predict(x)


### END CODE HERE
