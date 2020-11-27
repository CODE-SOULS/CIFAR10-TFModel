   # def build_network(self):
    #     inputs = layers.Input(shape=(32, 32, 3))
    #     x = layers.Conv2D(
    #         16, 3, input_shape=(32, 32, 3), padding="same", activation="relu"
    #     )(inputs)
    #     x = layers.Conv2D(32, 3, activation="relu")(x)
    #     x = layers.Flatten()(x)
    #     x = layers.Dense(128, activation="relu")(x)
    #     x = layers.Dense(10, activation="softmax")(x)
    #     return Model(inputs=inputs, outputs=x, name="mymodel")





# class MyNetwork(Model):
#     def __init__(
#         self,
#         configs,
#         dropout_rate=0.25,
#         num_classes=10,
#         num_layers=4,
#         increment=False,
#         increment_value=32,
#     ):
#         super(MyNetwork, self).__init__()
#         self.configs = configs
#         self._activation_fn = tf.nn.swish
#         self._dropout_rate = dropout_rate
#
#         if increment:
#             self._filters = [
#                 n
#                 for n in range(
#                     increment_value, increment_value * (num_layers + 1), increment_value
#                 )
#             ]
#         else:
#             self._filters = [
#                 n
#                 for n in reversed(
#                     range(
#                         increment_value,
#                         increment_value * (num_layers + 1),
#                         increment_value,
#                     )
#                 )
#             ]
#         # feature extraction layers
#         dropout_rate = 0.2
#         for i, num_filters in enumerate(self._filters):
#             setattr(
#                 self,
#                 f"conv_{num_filters}",
#                 self.conv_block(num_filters, kernel_size=3, dropout_rate=dropout_rate),
#             )
#             dropout_rate += 0.1
#             dropout_rate = round(dropout_rate, 2)
#         # classifier
#         self._classifier = Sequential(
#             [
#                 layers.Dropout(0.2),
#                 layers.Flatten(),
#                 layers.Dense(512, activation="relu", kernel_initializer="he_uniform"),
#                 layers.BatchNormalization(),
#                 layers.Dropout(0.5),
#                 layers.Dense(
#                     num_classes,
#                     activation="softmax",
#                     name="probs",
#                     kernel_initializer="he_uniform",
#                 ),
#             ],
#             name="classifier",
#         )
#
#     def conv_block(self, num_filters, kernel_size=3, dropout_rate=0.2):
#         return Sequential(
#             [
#                 layers.Conv2D(
#                     num_filters,
#                     kernel_size=kernel_size,
#                     padding="same",
#                     kernel_initializer="he_uniform",
#                 ),
#                 #layers.ReLU(),
#                 layers.Activation(activation=self._activation_fn),
#                 layers.BatchNormalization(-1),
#                 layers.Conv2D(
#                     num_filters,
#                     kernel_size=kernel_size,
#                     padding="same",
#                     kernel_initializer="he_uniform",
#                 ),
#                 #layers.ReLU(),
#                 layers.Activation(activation=self._activation_fn),
#                 layers.BatchNormalization(-1),
#                 layers.MaxPool2D((2, 2)),
#                 #layers.AveragePooling2D((2,2)),
#                 layers.Dropout(dropout_rate),
#             ]
#         )
#
#     def call(self, inputs, training=None, mask=None):
#         """
#         Args:
#             inputs: A Tensor representing a batch of input images.
#             training: A boolean. Used by operations that work differently
#                 in training and testing phases such as batch normalization.
#         Return:
#             The output Tensor of the network.
#         """
#         return self.build_network(inputs, training)
#
#     def build_network(self, x, training):
#         # x = self._stem(inputs)
#         # x = self.fe(x)
#         # x = self._top(x)
#         for num_filters in self._filters:
#             x = getattr(self, f"conv_{num_filters}")(x)
#         x = self._classifier(x)
#         return x
#
#
# ### END CODE HERE


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
        self.model = None

    @classmethod
    def augment_dataset(cls, dataset):
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        data_augmentation = tf.keras.Sequential([
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(factor=0.2),
                layers.experimental.preprocessing.RandomZoom(width_factor=0.15, height_factor=0.15)])
        dataset = dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=AUTOTUNE,
        )
        return dataset.prefetch(buffer_size=AUTOTUNE)

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
        #initial_learning_rate = 0.1
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
        # )
        #optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
        #optimizer = tf.keras.optimizers.Adam(lr=0.001,decay=0, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        optimizer = tf.keras.optimizers.Adam()
        self.model = self.network()
        self.model.compile(
            optimizer=optimizer, loss=loss, metrics=["sparse_categorical_accuracy"]
        )
        #steps = int(x_train.shape[0] / 64)
        #self.network.fit(train_dataset,validation_data=val_dataset, epochs=10, batch_size=32, verbose=2, callbacks=[self.model_checkpoint_callback])
        #self.network.summary()
        self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            #steps_per_epoch=steps,
            epochs=300,
            batch_size=128,
            verbose=2,
        )
        saved_model_folder = os.path.abspath(self.configs["save_dir"])
        self.model.save(saved_model_folder, overwrite=True, save_format="tf")

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

@classmethod
def augment_dataset(cls, dataset):
    # AUTOTUNE = tf.data.experimental.AUTOTUNE
    # data_augmentation = tf.keras.Sequential([
    #         layers.experimental.preprocessing.RandomFlip("horizontal"),
    #         layers.experimental.preprocessing.RandomRotation(factor=8.0)
    #     ]
    # )
    # ds = ds.map(
    #     lambda x, y: (data_augmentation(x, training=True), y),
    #     num_parallel_calls=AUTOTUNE,
    # )
    # # Use buffered prefecting on all datasets
    # return ds.prefetch(buffer_size=AUTOTUNE)
    # Add augmentations
    augmentations = [cls.flip, cls.color, cls.rotate]

    # Add the augmentations to the dataset
    for f in augmentations:
        # Apply the augmentation, run 4 jobs in parallel.
        dataset = dataset.map(lambda x, y: (f(x), y), num_parallel_calls=4)

    # Make sure that the values are still in [0, 1]
    dataset = dataset.map(
        lambda x, y: (tf.clip_by_value(x, 0, 1), y), num_parallel_calls=4
    )
    return dataset


@staticmethod
def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x


@staticmethod
def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x


@staticmethod
def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(
        x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    )
