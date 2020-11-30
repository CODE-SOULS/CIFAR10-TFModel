### YOUR CODE HERE

# import torch
import os

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from ImageUtils import parse_record
from Network import MyNetwork

"""This script defines the training, validation and testing process.
"""


class MyModel(object):
    def __init__(self, configs):
        self.configs = configs
        self.network = MyNetwork(configs)
        self.model = None

    def train(self, x_train, y_train, training_config, x_valid=None, y_valid=None):
        # get the training config parameters
        num_epochs = training_config["num_epochs"]
        batch_size = training_config["batch_size"]
        verbose = training_config["verbose"]
        learning_rate = training_config["learning_rate"]
        momentum = training_config["momentum"]

        # run processing in both train and validation dataset
        x_train = np.apply_along_axis(
            func1d=parse_record, arr=x_train, axis=1, training=True
        )
        x_valid = np.apply_along_axis(
            func1d=parse_record, arr=x_valid, axis=1, training=True
        )
        num_steps = int(x_train.shape[0] / batch_size)

        # configure optimizer, loss and accuracy function
        optimizer = tf.keras.optimizers.SGD(lr=learning_rate, momentum=momentum)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

        # create keras model instance
        self.model = self.network()

        # compile model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[acc_metric])

        # setup data augmentation
        datagen = ImageDataGenerator(
            width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True
        )

        # train model
        history = self.model.fit(
            datagen.flow(x_train, y_train, batch_size=batch_size),
            steps_per_epoch=num_steps,
            epochs=num_epochs,
            validation_data=(x_valid, y_valid),
            verbose=verbose,
        )
        # export trained model
        saved_model_folder = os.path.abspath(self.configs["save_dir"])
        self.model.save(saved_model_folder, overwrite=True, save_format="tf")
        return history

    def evaluate(self, x, y, load_saved_model=True):

        # load the model from disk if it's required
        if load_saved_model:
            saved_model_folder = os.path.abspath(self.configs["save_dir"])
            if os.path.exists(saved_model_folder):
                self.model = tf.keras.models.load_model(saved_model_folder)
        assert self.model, "model don't load"

        # processing step
        x = np.apply_along_axis(func1d=parse_record, arr=x, axis=1, training=True)

        loss, accc = self.model.evaluate(x, y)
        print(
            f"validation done : the loss and the accuracy reported by the model were {loss} and {accc}"
        )

    def predict_prob(self, x, load_saved_model=True):

        # load the model from disk if it's required
        if load_saved_model:
            saved_model_folder = os.path.abspath(self.configs["save_dir"])
            if os.path.exists(saved_model_folder):
                self.model = tf.keras.models.load_model(saved_model_folder)
        assert self.model, "model don't load"

        # processing step
        x = np.apply_along_axis(func1d=parse_record, arr=x, axis=1, training=False)

        return self.model.predict(x)


### END CODE HERE
