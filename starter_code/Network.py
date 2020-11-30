### YOUR CODE HERE
from tensorflow.keras import Model
from tensorflow.keras import layers

# import torch

"""This script defines the network.
"""


class MyNetwork(object):
    def __init__(self, configs):
        self.configs = configs

    def __call__(self, *args, **kwargs):
        return self.build_network()

    @staticmethod
    def feature_extraction_block(
        x,
        num_filters,
        kernel_size=(3, 3),
        activation_function="relu",
        kernel_initializer="he_uniform",
        padding="same",
        dropout_rate=0.2,
    ):
        x = layers.Conv2D(
            num_filters,
            kernel_size,
            activation=activation_function,
            kernel_initializer=kernel_initializer,
            padding=padding,
        )(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.Conv2D(
            num_filters,
            kernel_size,
            activation=activation_function,
            kernel_initializer=kernel_initializer,
            padding=padding,
        )(x)
        x = layers.BatchNormalization(axis=-1)(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout_rate)(x)
        return x

    @staticmethod
    def classifier_block(x, num_neurons, num_classes, dropout_rate=0.5):
        x = layers.Flatten()(x)
        x = layers.Dense(
            num_neurons, activation="relu", kernel_initializer="he_uniform"
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(num_classes, activation="softmax")(x)
        return x

    def build_network(self):
        input_shape = (32, 32, 3)
        num_classes = 10
        # define the model input and first CONV module
        inputs = layers.Input(shape=input_shape)
        x = self.feature_extraction_block(inputs, num_filters=32, dropout_rate=0.2)
        x = self.feature_extraction_block(x, num_filters=64, dropout_rate=0.3)
        x = self.feature_extraction_block(x, num_filters=128, dropout_rate=0.4)
        x = self.classifier_block(x, num_neurons=128, num_classes=num_classes)
        model = Model(inputs, x, name="final_model")
        return model
