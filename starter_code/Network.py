### YOUR CODE HERE
import math
from tensorflow.keras import Model, Sequential
from tensorflow.keras import layers
import tensorflow as tf

# import torch

"""This script defines the network.
"""


class MyNetwork(object):
    def __init__(self, configs):
        self.configs = configs

    def __call__(self, *args, **kwargs):
        return self.build_network()

    def conv_module(self, x, K, kX, kY, stride, chanDim, padding="same"):
        # define a CONV => BN => RELU pattern
        x = layers.Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Activation("relu")(x)
        # return the block
        return x

    def inception_module(self, x, numK1x1, numK3x3, chanDim):
        # define two CONV modules, then concatenate across the
        # channel dimension
        conv_1x1 = self.conv_module(x, numK1x1, 1, 1, (1, 1), chanDim)
        conv_3x3 = self.conv_module(x, numK3x3, 3, 3, (1, 1), chanDim)
        x = layers.concatenate([conv_1x1, conv_3x3], axis=chanDim)
        # return the block
        return x

    def downsample_module(self, x, K, chanDim):
        # define the CONV module and POOL, then concatenate
        # across the channel dimensions
        conv_3x3 = self.conv_module(x, K, 3, 3, (2, 2), chanDim, padding="valid")
        pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate([conv_3x3, pool], axis=chanDim)
        # return the block
        return x

    def build_network(self):
        chanDim = -1
        inputShape = (32, 32, 3)
        classes = 10
        # define the model input and first CONV module
        inputs = layers.Input(shape=inputShape)
        x = self.conv_module(inputs, 96, 3, 3, (1, 1), chanDim)
        # two Inception modules followed by a downsample module
        x = self.inception_module(x, 32, 32, chanDim)
        x = self.inception_module(x, 32, 48, chanDim)
        x = self.downsample_module(x, 80, chanDim)
        # four Inception modules followed by a downsample module
        x = self.inception_module(x, 112, 48, chanDim)
        x = self.inception_module(x, 96, 64, chanDim)
        x = self.inception_module(x, 80, 80, chanDim)
        x = self.inception_module(x, 48, 96, chanDim)
        x = self.downsample_module(x, 96, chanDim)
        # two Inception modules followed by global POOL and dropout
        x = self.inception_module(x, 176, 160, chanDim)
        x = self.inception_module(x, 176, 160, chanDim)
        x = layers.AveragePooling2D((7, 7))(x)
        x = layers.Dropout(0.5)(x)
        # softmax classifier
        x = layers.Flatten()(x)
        x = layers.Dense(classes)(x)
        x = layers.Activation("softmax")(x)
        # create the model
        model = Model(inputs, x, name="minigooglenet")
        # return the constructed network architecture
        return model
