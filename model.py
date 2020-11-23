import collections

import math

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, experimental
from tensorflow.keras import layers
from tensorflow.keras import *
from tensorflow import keras
from tensorflow.keras import Model
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pylab as plt
import random
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.python.keras.layers import ZeroPadding2D


class SakibNet(Model):
    def __init__(self):
        super(SakibNet, self).__init__()
        self.conv1 = Conv2D(
            16, 3, input_shape=(32, 32, 3), padding="same", activation="relu"
        )
        self.efcnet = EfficientNetB7()
        self.conv2 = Conv2D(32, 3, activation="relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="softmax")

    def call(self, x, training=None, mask=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


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

BlockArgs = collections.namedtuple(
    "BlockArgs",
    [
        "kernel_size",
        "num_repeat",
        "input_filters",
        "output_filters",
        "expand_ratio",
        "id_skip",
        "strides",
        "se_ratio",
    ],
)

ARCH_BLOCKS_ARGS = [
    BlockArgs(
        kernel_size=3,
        num_repeat=1,
        input_filters=32,
        output_filters=16,
        expand_ratio=1,
        id_skip=True,
        strides=[1, 1],
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=3,
        num_repeat=2,
        input_filters=16,
        output_filters=24,
        expand_ratio=6,
        id_skip=True,
        strides=[2, 2],
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=5,
        num_repeat=2,
        input_filters=24,
        output_filters=40,
        expand_ratio=6,
        id_skip=True,
        strides=[2, 2],
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=3,
        num_repeat=3,
        input_filters=40,
        output_filters=80,
        expand_ratio=6,
        id_skip=True,
        strides=[2, 2],
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=5,
        num_repeat=3,
        input_filters=80,
        output_filters=112,
        expand_ratio=6,
        id_skip=True,
        strides=[1, 1],
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=5,
        num_repeat=4,
        input_filters=112,
        output_filters=192,
        expand_ratio=6,
        id_skip=True,
        strides=[2, 2],
        se_ratio=0.25,
    ),
    BlockArgs(
        kernel_size=3,
        num_repeat=1,
        input_filters=192,
        output_filters=320,
        expand_ratio=6,
        id_skip=True,
        strides=[1, 1],
        se_ratio=0.25,
    ),
]
BATCH_SIZE = 32


class EfficientNet(Model):
    def __init__(self, dropout_rate=0.5, num_classes=10, version=6):
        super(EfficientNet, self).__init__()
        self._block_args: BlockArgs = ARCH_BLOCKS_ARGS[version]
        self._activation_fn = tf.nn.swish
        self._dropout_rate = dropout_rate
        # create the stem
        self._stem = Sequential(
            [
                layers.Conv2D(
                    filters=32,
                    kernel_size=3,
                    strides=(2, 2),
                    padding="same",
                    use_bias=False,
                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                ),
                layers.BatchNormalization(),
                layers.Activation(self._activation_fn),
            ],
            name="stem",
        )
        self._top = Sequential(
            [
                layers.Conv2D(
                    filters=1280,
                    kernel_size=1,
                    padding="same",
                    use_bias=False,
                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                ),
                layers.BatchNormalization(),
                layers.Activation(self._activation_fn),
            ],
            name="top",
        )
        self._classifier = Sequential(
            [
                layers.GlobalAveragePooling2D(),
                layers.Dropout(dropout_rate),
                layers.Dense(
                    num_classes,
                    activation="softmax",
                    kernel_initializer=DENSE_KERNEL_INITIALIZER,
                    name="probs",
                ),
            ],
            name="classifier",
        )
        self._m1 = self.module_1()
        self._m2 = self.module_2()
        # self._m3 = self.module_3()
        # self._m4 = self.module_4()
        # self._m5 = self.module_5()

    def module_1(self):
        return Sequential([
                layers.DepthwiseConv2D(
                    self._block_args.kernel_size,
                    strides=self._block_args.strides,
                    padding="same",
                    use_bias=False,
                    depthwise_initializer=CONV_KERNEL_INITIALIZER,
                ),
                layers.BatchNormalization(),
                layers.Activation(self._activation_fn),
                # new
                layers.Dropout(0.25)
        ]
        )

    def module_2(self):
        return Sequential([self.module_1(), ZeroPadding2D((1,1)), self.module_1()])

    def module_3(self):
        return Sequential([
            layers.GlobalAveragePooling2D(),
            layers.Reshape(target_shape=(1, 1, 32)),
            layers.Conv2D(
                filters=self._block_args.input_filters/2,
                activation=self._activation_fn,
                kernel_size=1,
                padding="same",
                use_bias=True,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
            ),
            layers.Conv2D(
                filters=self._block_args.input_filters,
                activation="sigmoid",
                kernel_size=1,
                padding="same",
                use_bias=True,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
            )
        ])

    def module_4(self):
        return Sequential([
            #layers.multiply([self.outputs[0] , self.outputs[-1]]),
            layers.Conv2D(
                filters=self._block_args.input_filters,
                kernel_size=1,
                padding="same",
                use_bias=False,
                kernel_initializer=CONV_KERNEL_INITIALIZER,
            ),
            layers.BatchNormalization()
        ])

    def module_5(self):
        return Sequential([
            self.module_4(),
            layers.Dropout(self._dropout_rate)
        ])

    def call(self, x, training=None, mask=None):
        x = self._stem(x)
        x = self._m1(x)
        x = self._m2(x)
        # x = self._m3(x)
        # x = self._m4(x)
        # x = self._m5(x)
        x = self._top(x)
        x = self._classifier(x)
        return x


def visualize_data(data_path, n=10, rows=2):
    data = np.load(data_path)
    start_index = random.randint(0, data.shape[0])
    end_index = start_index + n
    print(data.shape)
    images = data[start_index:end_index]
    fig: plt.Figure = plt.figure(figsize=(20, 20))
    cols = math.ceil(n / rows)
    for i, img in enumerate(images):
        ax = fig.add_subplot(rows, cols, i + 1)
        print(np.min(img), np.max(img))
        ax.imshow(img)
    plt.show()


AUTOTUNE = tf.data.experimental.AUTOTUNE


def prepare(ds, batch_size=32, shuffle=False):
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
    if shuffle:
        ds = ds.shuffle(1000)
    # create batches
    ds = ds.batch(batch_size)
    # Use buffered prefecting on all datasets
    return ds.prefetch(buffer_size=AUTOTUNE)


def load_dataset(batch_size=12):
    # Construct a tf.data.Dataset
    (ds_train, ds_validation), ds_info = tfds.load(
        "cifar10",
        split=["train[:80%]", "train[80%:]"],
        # shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    # num_classes = ds_info.features["label"].num_classes
    # classes_names = ds_info.features["label"].names
    # print(ds_info.splits["train"].num_examples)
    # fig = tfds.show_examples(ds_train, ds_info, rows = 10,cols= 10, plot_scale = 3.0)
    # samples = ds_train.take(4)
    # for i, (img, label) in enumerate(tfds.as_numpy(samples)):
    #     print(img.shape, label)
    ds_train = prepare(ds_train, batch_size=batch_size, shuffle=True)
    ds_validation = prepare(ds_validation, batch_size=batch_size)
    return ds_train, ds_validation


if __name__ == "__main__":
    assert len(tf.config.list_physical_devices("GPU")) > 0, "Not GPU detected"
    # visualize_data("data/private_test_images.npy", n = 50, rows=5)
    ds_train, ds_validation = load_dataset()
    model = EfficientNet(num_classes=10, version=6)
    # model.summary()
    # for val_images, val_labels in ds_train.take(3):
    #     model(val_images, val_labels)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    #optimizer = tf.keras.optimizers.SGD()
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")
    val_loss = tf.keras.metrics.Mean(name="test_loss")
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="test_accuracy")

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def val_step(images, labels):
        predictions = model(images)
        t_loss = loss_object(labels, predictions)
        val_loss(t_loss)
        val_accuracy(labels, predictions)

    EPOCHS = 100
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        for images, labels in ds_train:
            train_step(images, labels)

        for val_images, val_labels in ds_validation:
            val_step(val_images, val_labels)

        template = "Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}"
        print(
            template.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result() * 100,
                val_loss.result(),
                val_accuracy.result() * 100,
            )
        )
