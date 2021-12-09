# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:38:15 2021

@author: Sokratis Koseoglou
"""

import tensorflow as tf
from tensorflow import keras
# import datetime
# from  tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras import layers
# import os

tf.debugging.set_log_device_placement(True)

# os.path.abspath(os.getcwd())
# os.chdir('./Desktop/Thesis/Codes')

from Xception import myXception
# from mySimpleCNN import simpleCNN

dataset_dir = "2016_Copy_2"

image_size = (240, 240)
batch_size = 32
epochs = 1
seed = 100 # 1337

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split = 0.2,
    subset = "training",
    seed = seed,
    image_size = image_size,
    batch_size = batch_size,
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = seed,
    image_size = image_size,
    batch_size = batch_size,
)

print('\n------------------------------------------------------------------------------------------------------\n')

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

# with strategy.scope():
    
# model = simpleCNN(input_shape = image_size + (3,))
model = myXception(input_shape = image_size + (3,))
    
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
    

model.fit(
    train_ds, epochs = epochs, callbacks = callbacks, validation_data = test_ds
)



# (eval_loss, eval_accuracy) = model.evaluate(test_ds, batch_size = batch_size, verbose=1)

# print('[INFO] accuracy: {:.2f}%'.format(eval_accuracy * 100)) 
# print('[INFO] Loss: {}'.format(eval_loss)) 