import math

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, experimental
from tensorflow.keras import layers
from tensorflow.keras import Model
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pylab as plt
import random


class SakibNet(Model):
    def __init__(self):
        super(SakibNet, self).__init__()
        self.conv1 = Conv2D(16, 3, input_shape=(32, 32, 3), padding="same", activation="relu")
        self.conv2 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="softmax")

    def call(self, x, training=None, mask=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


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
    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y), num_parallel_calls=AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    # create batches
    ds = ds.batch(batch_size)
    # Use buffered prefecting on all datasets
    return ds.prefetch(buffer_size=AUTOTUNE)


def load_dataset(batch_size=32):
    # Construct a tf.data.Dataset
    (ds_train, ds_validation), ds_info = tfds.load(
        "cifar10",
        split=["train[:80%]", "train[80%:]"],
        #shuffle_files=True,
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

    model = SakibNet()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
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

    EPOCHS = 50
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        for images, labels in ds_train:
            train_step(images, labels)

        for val_images, val_labels in ds_validation:
            val_step(val_images, val_labels)

        template = "Epoch {}, Loss: {0:2}, Accuracy: {0:2}, Val Loss: {0:2}, Val Accuracy: {0:2}"
        print(
            template.format(
                epoch + 1,
                train_loss.result(),
                train_accuracy.result() * 100,
                val_loss.result(),
                val_accuracy.result() * 100,
            )
        )
