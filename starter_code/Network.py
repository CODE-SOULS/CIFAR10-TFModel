### YOUR CODE HERE
from tensorflow.keras import Model
from tensorflow.keras import layers
# import torch

"""This script defines the network.
"""

class MyNetwork(Model):

    def __init__(self, configs):
        super(MyNetwork, self).__init__()
        self.configs = configs
        self.conv1 = layers.Conv2D(
            16, 3, input_shape=(32, 32, 3), padding="same", activation="relu"
        )
        self.conv2 = layers.Conv2D(32, 3, activation="relu")
        self.flatten = layers.Flatten()
        self.d1 = layers.Dense(128, activation="relu")
        self.d2 = layers.Dense(10, activation="softmax")

    def call(self, inputs, training=None, mask=None):
        '''
        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Used by operations that work differently
                in training and testing phases such as batch normalization.
        Return:
            The output Tensor of the network.
        '''
        return self.build_network(inputs, training)

    def build_network(self, inputs, training):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


### END CODE HERE