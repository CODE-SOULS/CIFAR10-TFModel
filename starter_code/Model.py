### YOUR CODE HERE

# import torch
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from ImageUtils import parse_record
from Network import MyNetwork

"""This script defines the training, validation and testing process.
"""


class MyModel(object):
    def __init__(self, configs):
        self.configs = configs
        self.network = MyNetwork(configs)
        self.model = None

    @classmethod
    def augment_dataset(cls, dataset):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data_augmentation = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(factor=0.2),
                layers.experimental.preprocessing.RandomZoom(
                    width_factor=0.15, height_factor=0.15
                ),
            ]
        )
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
        return dataset.prefetch(buffer_size=AUTOTUNE)

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        num_epochs = configs["num_epochs"]
        batch_size = configs["batch_size"]
        verbose = configs["verbose"]

        x_train = np.apply_along_axis(
            func1d=parse_record, arr=x_train, axis=1, training=True
        )
        x_valid = np.apply_along_axis(
            func1d=parse_record, arr=x_valid, axis=1, training=True
        )
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(32)

        train_dataset = self.augment_dataset(train_dataset)

        # initial_learning_rate = 0.1
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
        # )
        # optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        # optimizer = tf.keras.optimizers.Adam(lr=0.001,decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        optimizer = tf.keras.optimizers.Adam()
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.model = self.network()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
        self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=num_epochs,
            batch_size=batch_size,
            verbose=verbose,
        )
        saved_model_folder = os.path.abspath(self.configs["save_dir"])
        self.model.save(saved_model_folder, overwrite=True, save_format="tf")

    def evaluate(self, x, y):
        x = np.apply_along_axis(func1d=parse_record, arr=x, axis=1, training=True)
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
