"""
author: Sanidhya Mangal
github: sanidhyamangal
"""
import os  # for os related ops
from functools import partial  # for creation of partial functions

import tensorflow as tf  # for deep learning based options


class GenerativeModel(tf.keras.Model):
    """
    Generative model for conditional gans
    """
    def __init__(self, *args, **kwargs):
        super(GenerativeModel, self).__init__(*args, **kwargs)

        # create partial methods for conv model
        Conv2DT = partial(tf.keras.layers.Conv2DTranspose,
                          kernel_size=(5, 5),
                          padding="same",
                          strides=(2, 2),
                          use_bias=False)
        BN = partial(tf.keras.layers.BatchNormalization)
        LRU = partial(tf.keras.layers.LeakyReLU)

        # create a genrative model
        self.dense1 = tf.keras.layers.Dense(7 * 7 * 128, use_bias=False)
        self.conditional_dense = tf.keras.layers.Dense(units=7 * 7 * 128)

        self.bn1 = BN()
        self.act1 = LRU()

        self.reshape = tf.keras.layers.Reshape((7, 7, 128))

        self.deconv1 = Conv2DT(filters=128)
        self.bn2 = BN()
        self.act2 = LRU()

        self.deconv2 = Conv2DT(filters=1)
        self.bn3 = BN()
        self.act3 = LRU()

    def call(self, inputs, labels, *args, **kwargs):
        training = kwargs.pop('training', True)

        # combine labels and inputs
        x = tf.concat(
            [inputs,
             self.conditional_dense(tf.one_hot(labels, depth=10))],
            axis=1)
        x = self.dense1(inputs)

        x = self.act1(self.bn1(x, training=training))
        x = self.reshape(x)

        # first deconv and upscaling part
        x = self.deconv1(x)
        x = self.act2(self.bn2(training=training))

        # second deconv and upscaling layer
        x = self.deconv2(x)
        x = self.act3(self.bn3(x, training=training))

        return x


class DiscrimintaiveModel(tf.keras.models.Model):
    """
    Class for perfroming discriminative models for the conditional lsgan
    """
    def __init__(self, *args, **kwargs):
        super(DiscrimintaiveModel, self).__init__(*args, **kwargs)

        # create partial methods for model
        Conv2D = partial(tf.keras.layers.Conv2D,
                         kernel_size=(5, 5),
                         padding="same",
                         strides=(2, 2),
                         use_bias=False)
        FC = partial(tf.keras.layers.Dense)
        BN = partial(tf.keras.layers.BatchNormalization)
        self.conditional_dense = FC(units=256)
        self.conv1 = Conv2D(filters=256)

        self.conv2 = Conv2D(filters=320)
        self.bn1 = BN()

        self.fc1 = FC(units=1024)
        self.bn2 = BN()
        self.fc2 = FC(units=1)

    def call(self, inputs, labels, *args, **kwargs):
        training = kwargs.pop('training', True)

        conditional_vector = self.conditional_dense(
            tf.one_hot(labels, depth=10))
        conditional_vector_conv = tf.reshape(conditional_vector,
                                             [-1, 1, 1, 256])
        x = self.conv1(inputs)
        x = self.conv_concat(x, conditional_vector_conv)

        # conv layer ops 2
        x = self.bn1(self.conv2(x), training=training)
        x = tf.reshape(x, [x.shape[0], -1])

        x = tf.concat([x, conditional_vector], axis=1)

        x = self.bn2(self.fc1(x), training=training)

        x = self.fc2(x)
        return x

    @tf.function
    def conv_concat(self, x: tf.Tensor, y: tf.Tensor):
        x_shape = x.shape
        y_shape = y.shape

        return tf.concat([x, y * tf.ones(shape=x_shape)], axis=3)
