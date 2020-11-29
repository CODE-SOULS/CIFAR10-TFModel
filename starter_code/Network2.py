### YOUR CODE HERE
import math
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB7, DenseNet121
import tensorflow as tf

# import torch

"""This script defines the network.
"""

CONV_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        "distribution": "normal",
    },
}

DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 1.0 / 3.0, "mode": "fan_out", "distribution": "uniform"},
}


class MyNetwork(object):
    def __init__(self, configs):
        self.configs = configs

    def __call__(self, *args, **kwargs):
        return self.build_network()

    def conv_block(self, num_filters, kernel_size=3, dropout_rate=0.2):
        return Sequential(
            [
                layers.Conv2D(
                    num_filters,
                    kernel_size=kernel_size,
                    padding="same",
                    kernel_initializer=tf.initializers.variance_scaling(),
                ),
                # layers.Activation(tf.nn.swish),
                layers.ReLU(),
                layers.BatchNormalization(-1, momentum=0.997, epsilon=1e-5),
                layers.Conv2D(
                    num_filters,
                    kernel_size=kernel_size,
                    padding="same",
                    kernel_initializer=tf.initializers.variance_scaling(),
                ),
                layers.ReLU(),
                # layers.Activation(tf.nn.swish),
                layers.BatchNormalization(-1, momentum=0.997, epsilon=1e-5),
                layers.MaxPool2D((2, 2)),
                # layers.AveragePooling2D(),
                layers.Dropout(dropout_rate),
            ]
        )

    def head_block(self, num_classes):
        return Sequential(
            [
                layers.Dropout(0.5),
                layers.Flatten(),
                layers.Dense(512, activation="relu", kernel_initializer="he_uniform"),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(
                    num_classes,
                    activation="softmax",
                    kernel_initializer="he_uniform",
                ),
            ]
        )

    def build_network(self):
        input_shape = (32, 32, 3)
        classes = 10
        num_layers = 5
        filters = []
        inputs = layers.Input(shape=input_shape)

        x1 = self.conv_block(32, kernel_size=3, dropout_rate=0.2)(inputs)

        x2 = self.conv_block(32, kernel_size=3, dropout_rate=0.2)(inputs)

        x = layers.concatenate([x1, x2])

        x1 = self.conv_block(64, kernel_size=3, dropout_rate=0.3)(x)

        x2 = self.conv_block(64, kernel_size=3, dropout_rate=0.3)(x)

        x = layers.concatenate([x1, x2])

        x1 = self.conv_block(128, kernel_size=3, dropout_rate=0.5)(x)

        x2 = self.conv_block(128, kernel_size=3, dropout_rate=0.5)(x)

        x = layers.concatenate([x1, x2])

        # num_filters = 32
        # for _ in range(num_layers):
        #     filters.append(num_filters)
        #     num_filters *= 2
        # dropout_rate = 0.2
        # for i, num_filters in enumerate(filters):
        #     print(num_filters, dropout_rate)
        #     x = self.conv_block(
        #         num_filters,
        #         kernel_size=3,
        #         dropout_rate=dropout_rate
        #     )(inputs if i == 0 else x)
        #     dropout_rate += 0.1
        #     dropout_rate = round(dropout_rate, 2)

        x = self.head_block(classes)(x)
        model = Model(inputs, x, name="HenryNet")
        return model
