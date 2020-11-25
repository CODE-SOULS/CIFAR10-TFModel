### YOUR CODE HERE
import math
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers
import tensorflow as tf

# import torch

"""This script defines the network.
"""


class MyNetwork(Model):
    def __init__(
        self,
        configs,
        dropout_rate=0.25,
        num_classes=10,
        num_layers=5,
        increment=False,
        increment_value=32,
    ):
        super(MyNetwork, self).__init__()
        self.configs = configs
        self._activation_fn = tf.nn.swish
        self._dropout_rate = dropout_rate

        if increment:
            self._filters = [
                n
                for n in range(
                    increment_value, increment_value * (num_layers + 1), increment_value
                )
            ]
        else:
            self._filters = [
                n
                for n in reversed(
                    range(
                        increment_value,
                        increment_value * (num_layers + 1),
                        increment_value,
                    )
                )
            ]
        # feature extraction layers
        dropout_rate = 0.2
        for i, num_filters in enumerate(self._filters):
            setattr(
                self,
                f"conv_{num_filters}",
                self.conv_block(num_filters, kernel_size=3, dropout_rate=dropout_rate),
            )
            dropout_rate += 0.1
            dropout_rate = round(dropout_rate, 2)
        # classifier
        self._classifier = Sequential(
            [
                layers.Flatten(),
                layers.Dense(512, activation="relu", kernel_initializer="he_uniform"),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(
                    num_classes,
                    activation="softmax",
                    name="probs",
                    kernel_initializer="he_uniform",
                ),
            ],
            name="classifier",
        )

    def conv_block(self, num_filters, kernel_size=3, dropout_rate=0.2):
        return Sequential(
            [
                layers.Conv2D(
                    num_filters,
                    kernel_size=kernel_size,
                    padding="same",
                    kernel_initializer="he_uniform",
                ),
                #layers.ReLU(),
                layers.Activation(activation=self._activation_fn),
                layers.BatchNormalization(-1),
                layers.Conv2D(
                    num_filters,
                    kernel_size=kernel_size,
                    padding="same",
                    kernel_initializer="he_uniform",
                ),
                #layers.ReLU(),
                layers.Activation(activation=self._activation_fn),
                layers.BatchNormalization(-1),
                layers.MaxPool2D((2, 2)),
                #layers.AveragePooling2D((2,2)),
                layers.Dropout(dropout_rate),
            ]
        )

    def call(self, inputs, training=None, mask=None):
        """
        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Used by operations that work differently
                in training and testing phases such as batch normalization.
        Return:
            The output Tensor of the network.
        """
        return self.build_network(inputs, training)

    def build_network(self, x, training):
        # x = self._stem(inputs)
        # x = self.fe(x)
        # x = self._top(x)
        for num_filters in self._filters:
            x = getattr(self, f"conv_{num_filters}")(x)
        x = self._classifier(x)
        return x


### END CODE HERE
