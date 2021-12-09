# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:38:15 2021

@author: Sokratis Koseoglou
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Flatten, Dense
from keras.models import Model
# import os

# tf.debugging.set_log_device_placement(True)

# os.path.abspath(os.getcwd())
# os.chdir('./Desktop/Thesis/Codes')

dataset_dir = "2016_Copy_2"

image_size = (240, 240)
batch_size = 32
epochs = 1
# seed = 100 # 1337

# f = [0 for i in range(5)]
# seed_list = [100]

# for seed in seed_list:

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split = 0.2,
    subset = "training",
    seed = 100,
    image_size = image_size,
    batch_size = batch_size,
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 100,
    image_size = image_size,
    batch_size = batch_size,
)

print('\n------------------------------------------------------------------------------------------------------\n')

# strategy = tf.distribute.MirroredStrategy()
# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

print('\n------------------------------------------------------------------------------------------------------\n')

# with strategy.scope():

# add preprocessing layer to the front of VGG
m = ResNet50(input_shape = image_size + (3,), weights='imagenet', include_top=False)

# don't train existing weights
for layer in m.layers:
  layer.trainable = False

# our layers - you can add more if you want
x = Flatten()(m.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(units = 1, activation='sigmoid')(x)

# create a model object
model = Model(inputs = m.input, outputs = prediction)

# view the structure of the model
# model.summary()
                
callbacks = [
     keras.callbacks.ModelCheckpoint(
     filepath='model_{epoch}',
     save_freq='epoch')
]
    
model.compile(
    optimizer = keras.optimizers.Adam(1e-3),
    loss = "binary_crossentropy",
    metrics = ["accuracy"],
)

f = model.fit(
  train_ds,
  validation_data = test_ds,
  epochs = epochs,
  steps_per_epoch = len(train_ds),
  validation_steps = len(test_ds)
)

# sum = 0

# for i in range(len(seed_list)):
#     sum = sum + f[i].history['accuracy']

# print(sum)