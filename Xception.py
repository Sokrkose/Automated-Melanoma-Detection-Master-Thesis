# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:37:43 2021

@author: Sokratis Koseoglou
"""

# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# data_augmentation = tf.keras.Sequential([
#   layers.RandomFlip("horizontal_and_vertical"),
#   layers.RandomRotation(0.2),
#   layers.RandomZoom(0.2),
#   layers.GaussianNoise(1.0, seed = 1337)
# ])

def myXception(input_shape):
    
    inputs = keras.Input(shape=input_shape)
    
    # # Image augmentation block
    # x = data_augmentation(inputs)

    # Entry block
    
    x = layers.Rescaling(1.0 / 255)(inputs)
    
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units = 1, activation="sigmoid")(x)
    
    return keras.Model(inputs, outputs)