"""
author: Sanidhya Mangal
github: sanidhyamangal
"""
import os  # for os related ops
from functools import partial, reduce
from typing import Tuple  # for creation of partial functions

import tensorflow as tf  # for deep learning based options


class GenerativeModel(tf.keras.Model):
    """
    Generative model for conditional gans
    """
    def __init__(self, num_class:int=10, *args, **kwargs):
        super(GenerativeModel, self).__init__(*args, **kwargs)

        # create partial methods for conv model
        Conv2DT = partial(tf.keras.layers.Conv2DTranspose,
                          kernel_size=(5, 5),
                          padding="same",
                          strides=(2, 2),
                          use_bias=False)
        BN = partial(tf.keras.layers.BatchNormalization)
        LRU = partial(tf.keras.layers.LeakyReLU)

        # embedding layers and conditional model
        self.conditional = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(num_class, 50),
            tf.keras.layers.Dense(7*7),
            tf.keras.layers.Reshape((7,7,1))
        ])

        self.pre_gen = tf.keras.models.Sequential([
            tf.keras.layers.Dense(7*7*128),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Reshape((7,7,128))
        ])

        self.gen = tf.keras.models.Sequential([
            Conv2DT(filters=128, strides=(1,1)),
            BN(),
            LRU(),
            Conv2DT(filters=64),
            BN(),
            LRU(),
            Conv2DT(filters=1),
            BN(),
            LRU()
        ])



    def call(self, inputs, labels, *args, **kwargs):
        training = kwargs.pop('training', True)
        conditional_vec = self.conditional(labels)
        x = self.pre_gen(inputs)
        x = tf.concat([x, conditional_vec], axis=-1)
        x = self.gen(x, training=training)

        return x
        


class DiscrimintaiveModel(tf.keras.models.Model):
    """
    Class for perfroming discriminative models for the conditional lsgan
    """
    def __init__(self, input_shape:Tuple[int]=(28,28,1),num_classes:int=10,*args, **kwargs):
        super(DiscrimintaiveModel, self).__init__(*args, **kwargs)

        # create partial methods for model
        Conv2D = partial(tf.keras.layers.Conv2D,
                         kernel_size=(5, 5),
                         padding="same",
                         strides=(2, 2),
                         use_bias=False)
        FC = partial(tf.keras.layers.Dense)
        Relu = partial(tf.keras.layers.ReLU)
        self.conditional = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(num_classes, 50),
            tf.keras.layers.Dense(reduce(lambda a,b: a*b, input_shape)),
            tf.keras.layers.Reshape(input_shape)
        ])

        self.conv = tf.keras.models.Sequential([
            Conv2D(filters=32),
            Relu(),
            tf.keras.layers.Dropout(rate=0.3),
            Conv2D(filters=64),
            Relu(),
            tf.keras.layers.Dropout(rate=0.3),
            Conv2D(filters=64),
            Relu(),
            tf.keras.layers.Dropout(rate=0.3),
            tf.keras.layers.Flatten(),
            FC(1)
        ])

    def call(self, inputs, labels, *args, **kwargs):
        training = kwargs.pop('training', True)
        
        conditional_vector = self.conditional(labels)

        x = tf.concat([inputs, conditional_vector], axis=-1)
        x = self.conv(x)

        return x
