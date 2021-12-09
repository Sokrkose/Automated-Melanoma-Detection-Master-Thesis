# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 10:46:54 2021

@author: Sokratis Koseoglou
"""

import tensorflow as tf
from tensorflow import keras
from  tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras import layers
import os
# from keras.utils.vis_utils import plot_model

os.path.abspath(os.getcwd())
# os.chdir('./Desktop/Master Thesis/Codes')

from Xception import myXception
from mySimpleCNN import simpleCNN

train_dir = "../Datasets/2016/2016_Copy/train"
test_dir = "../Datasets/2016/2016_Copy/test"

image_size = (100, 100)
batch_size = 32

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_ds = train_datagen.flow_from_directory(
    train_dir,
    target_size = image_size,
    batch_size = batch_size,
    class_mode='binary'
)

test_ds = test_datagen.flow_from_directory(
    test_dir,
    target_size = image_size,
    batch_size = batch_size,
    class_mode='binary'
)

model = simpleCNN(input_shape = image_size + (3,))
# model = myXception(input_shape = image_size + (3,))

# plot_model(model, show_shapes=True)
# model.summary()

epochs = 1

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
]

model.compile(
    optimizer = keras.optimizers.Adam(1e-3),
    loss = "binary_crossentropy",
    metrics = ["accuracy"],
)

model.fit(
    train_ds, epochs = epochs, callbacks = callbacks, validation_data = test_ds
)

(eval_loss, eval_accuracy) = model.evaluate(test_ds, batch_size = batch_size, verbose=1)

print('[INFO] accuracy: {:.2f}%'.format(eval_accuracy * 100)) 
print('[INFO] Loss: {}'.format(eval_loss)) 