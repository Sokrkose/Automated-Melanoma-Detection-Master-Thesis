# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:38:15 2021

@author: Sokratis Koseoglou
"""

# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Flatten, Dense
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import statistics
from log import log

# tf.debugging.set_log_device_placement(True)

# import os
# os.path.abspath(os.getcwd())
# os.chdir('./Desktop/Thesis/Codes')

dataset_dir = "2016_Copy_2"
train_dir = "2016_Copy/train"
test_dir = "2016_Copy/test"

image_size = (224, 224)
batch_size = 32
epochs = 10

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(train_dir,
                                                 target_size = image_size,
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(test_dir,
                                            target_size = image_size,
                                            batch_size = batch_size,
                                            class_mode = 'binary')

print('\n\n')

# 'include top' is to determine whether or not to add the last layer of the model
m = VGG16(input_shape = image_size + (3, ), 
          weights = 'imagenet', 
          include_top = False)

model_name = "VGG16 with 3 more Dense Layers 1000 neurons each"

# don't train existing weights
for layer in m.layers:
  layer.trainable = False

# our layers - you can add more if you want
x = Flatten()(m.output)
x = Dense(1000, activation='relu')(x)
x = Dense(1000, activation='relu')(x)
x = Dense(1000, activation='relu')(x)
prediction = Dense(units = 1, activation='sigmoid')(x)

# create a model object
model = Model(inputs = m.input, outputs = prediction)

# view the structure of the model
model.summary()

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

m = model.fit(
  training_set,
  validation_data = test_set,
  epochs = epochs,
  steps_per_epoch = len(training_set),
  validation_steps = len(test_set)
)

acc = statistics.mean(m.history['val_accuracy'])

log(model_name, m.history['val_accuracy'], acc, epochs, batch_size, image_size)