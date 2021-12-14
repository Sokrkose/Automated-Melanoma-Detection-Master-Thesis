# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:38:15 2021

@author: Sokratis Koseoglou
"""

# import tensorflow as tf
from tensorflow import keras
from mySimpleCNN import simpleCNN
from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Flatten, Dense
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.metrics import classification_report
import numpy as np
import statistics
from log import log
import wandb
from wandb.keras import WandbCallback

# tf.debugging.set_log_device_placement(True)

# import os
# os.path.abspath(os.getcwd())
# os.chdir('./Desktop/Thesis/Codes')

train_dir = "2016_Copy/train"
test_dir = "2016_Copy/test"

image_size = (224, 224)
batch_size = 32
epochs = 2
learning_rate = 1e-3
optimizer = keras.optimizers.Adam(learning_rate)
model_name = "confusion matrix test"

wandb.init(project = "my-test-project", entity = "sokrkose", name = model_name)

wandb.config = {
    "image_size": image_size,
    "optimizer": optimizer,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size
}

train_data_generated = ImageDataGenerator(rescale = 1./255,
                                   samplewise_center = False,
                                   samplewise_std_normalization = False,
                                   rotation_range = 180,
                                   width_shift_range = [-30, 30],
                                   height_shift_range = [-30, 30],
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   fill_mode = 'nearest')

test_data_generated = ImageDataGenerator(rescale = 1./255,
                                  samplewise_center = False,
                                  samplewise_std_normalization = False)

training_set = train_data_generated.flow_from_directory(train_dir,
                                                 target_size = image_size,
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

test_set = test_data_generated.flow_from_directory(test_dir,
                                            target_size = image_size,
                                            batch_size = batch_size,
                                            class_mode = 'binary')

print('\n\n')

# 'include top' is to determine whether or not to add the last layer of the model
m = VGG16(include_top = True,
              weights = "imagenet",
             input_shape = image_size + (3,),
             classes = 1000,
             classifier_activation = "softmax")

# don't train existing weights
for layer in m.layers:
  layer.trainable = False
  
model = simpleCNN(image_size + (3,))

# our layers - you can add more if you want
# x = Flatten()(m.output)
# x = Dense(1024, activation='relu')(x)
# x = Dense(512, activation='relu')(x)
# x = Dense(256, activation='relu')(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
prediction = Dense(1, activation='sigmoid')(m.output)

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
    optimizer = optimizer,
    loss = "binary_crossentropy",
    metrics = ["accuracy"],
)

m = model.fit(
  training_set,
  validation_data = test_set,
  epochs = epochs,
  callbacks = [WandbCallback()],
  steps_per_epoch = len(training_set),
  validation_steps = len(test_set)
)

acc = statistics.mean(m.history['val_accuracy'])



test_steps_per_epoch = np.math.ceil(test_set.samples / test_set.batch_size)
predictions = model.predict(test_set, steps = test_steps_per_epoch)
# Get most likely class
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_set.classes
class_labels = list(test_set.class_indices.keys())   
# report = classification_report(true_classes, predicted_classes, target_names = class_labels)
# print(report)  

wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs = None,
                                                    preds = predicted_classes, 
                                                    y_true = true_classes,
                                                    class_names = class_labels)})

log(model_name, m.history['val_accuracy'], acc, epochs, batch_size, image_size, optimizer, learning_rate)