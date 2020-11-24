### YOUR CODE HERE
from pathlib import Path

import tensorflow as tf

# import torch
import os, time
import numpy as np
from Network import MyNetwork
from tensorflow.keras import layers
from ImageUtils import parse_record

"""This script defines the training, validation and testing process.
"""


class MyModel(object):
    def __init__(self, configs):
        self.configs = configs
        self.network = MyNetwork(configs)
        self.model_checkpoint_callback = None
        self.model_setup()

    def model_setup(self):
        saved_model = os.path.abspath(self.configs["save_dir"])
        os.makedirs(saved_model, exist_ok=True)
        checkpoint_filepath = os.path.join(saved_model, "checkpoint")
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor="val_sparse_categorical_accuracy",
            mode="max",
            save_best_only=True,
        )

    def augment_dataset(self, ds):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data_augmentation = tf.keras.Sequential(
            [
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                layers.experimental.preprocessing.RandomRotation(0.1),
                layers.experimental.preprocessing.RandomContrast(0.1),
            ]
        )
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
        # Use buffered prefecting on all datasets
        return ds.prefetch(buffer_size=AUTOTUNE)

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):

        x_train = np.apply_along_axis(
            func1d=parse_record, arr=x_train, axis=1, training=True
        )
        x_valid = np.apply_along_axis(
            func1d=parse_record, arr=x_valid, axis=1, training=True
        )

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(32)
        train_dataset = self.augment_dataset(train_dataset)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        initial_learning_rate = 0.1
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
        )
        #optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        optimizer = tf.keras.optimizers.Adam()
        self.network.compile(
            optimizer=optimizer, loss=loss, metrics=["sparse_categorical_accuracy"]
        )
        # self.network.fit(train_dataset,validation_data=val_dataset, epochs=10, batch_size=32, verbose=2, callbacks=[self.model_checkpoint_callback])
        self.network.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=400,
            batch_size=32,
            verbose=2,
        )
        saved_model_folder = os.path.abspath(self.configs["save_dir"])
        self.network.save(saved_model_folder, overwrite=True, save_format="tf")

    def evaluate(self, x, y):
        # saved_model = os.path.abspath(self.configs["save_dir"])
        # checkpoint_filepath = os.path.join(saved_model, 'checkpoint')        #
        # if os.path.exists(checkpoint_filepath):
        #     self.network.load_weights(checkpoint_filepath)
        # x = np.apply_along_axis(func1d=parse_record, arr=x, axis=1, training=True)
        # loss = tf.keras.losses.SparseCategoricalCrossentropy()
        # # optimizer = tf.keras.optimizers.SGD()
        # optimizer = tf.keras.optimizers.Adam()
        # self.network.compile(optimizer=optimizer,
        #                      loss=loss,
        #                      metrics=['sparse_categorical_accuracy'])

        # saved_model_folder = os.path.abspath(self.configs["save_dir"])
        # if os.path.exists(saved_model_folder):
        #     self.network = tf.keras.models.load_model(saved_model_folder)
        x = np.apply_along_axis(func1d=parse_record, arr=x, axis=1, training=True)

        loss, accc = self.network.evaluate(x, y)
        print(
            f"validation done : the loss and the accuracy reported by the model were {loss} and {accc}"
        )

    def predict_prob(self, x):
        saved_model_folder = os.path.abspath(self.configs["save_dir"])
        if os.path.exists(saved_model_folder):
            self.network = tf.keras.models.load_model(saved_model_folder)

        x = np.apply_along_axis(func1d=parse_record, arr=x, axis=1, training=False)
        return self.network.predict(x)


### END CODE HERE
