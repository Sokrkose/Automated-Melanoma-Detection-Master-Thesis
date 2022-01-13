# -*- coding: utf-8 -*-
"""
Created on Mon Jan  3 12:48:08 2022

@author: Sokratis Koseoglou
"""

import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import PIL
import cv2 as cv
import os

os.path.abspath(os.getcwd())
os.chdir('./Desktop/Thesis/Codes')

train_dir = "2016_Copy/train"
test_dir = "2016_Copy/test"
data_dir = "2016_Copy_2"
image_size = (224, 224)
batch_size = 16
epochs = 30
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
metrics = ['accuracy']
project_name = "test 3-1"
model_name = "efficientnet.EfficientNetB7"
entity = "sokrkose"

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split = 0.2,
    subset = 'training',
    seed = 123,
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size = image_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split = 0.2,
    subset = 'validation',
    seed = 123,
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size = image_size
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size = image_size
)







folder = '2016_Copy/test/1'
filename = 'ISIC_0000013.jpg'

img = cv.imread(os.path.join(folder, filename))
print(img.shape)

cv.imshow(filename, img)

width = int(img.shape[1] * 0.75)
height = int(img.shape[0] * 0.75)
print(width, height)

img = cv.resize(img, (224, 224), interpolation = cv.INTER_AREA)

cv.imshow(filename + " resized", img)
cv.waitKey(0)