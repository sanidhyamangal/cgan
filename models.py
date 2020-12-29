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

        self.concat = tf.keras.layers.Concatenate()

        self.reshape = tf.keras.layers.Reshape((7, 7, 128))

        self.deconv1 = Conv2DT(filters=128)
        self.bn2 = BN()
        self.act2 = LRU()

        self.deconv2 = Conv2DT(filters=1)
        self.bn3 = BN()
        self.act3 = LRU()

    def call(self, inputs, labels):
        x = tf.concat([inputs, self.conditional_dense(tf.one_hot(labels, depth=10))], axis=1)
        x = self.act1(self.bn1(self.dense1(inputs)))
        x = self.reshape(x)
        x = self.act2(self.bn2(self.deconv1(x)))
        x = self.act3(self.bn3(self.deconv2(x)))

        return x

