# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 11:49:35 2022

@author: Sokratis Koseoglou
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Dropout
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import statistics
from log import log
import wandb
from wandb import util
from wandb.keras import WandbCallback
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import PIL

# tf.debugging.set_log_device_placement(True)

# import os
# os.path.abspath(os.getcwd())
# os.chdir('./Desktop/Thesis/Codes')

train_dir = "2016_Copy/train"
test_dir = "2016_Copy/test"
data_dir = "2016_Copy_2"
image_size = (224, 224)
batch_size = 32
epochs = 30
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
metrics = ['accuracy']
project_name = "test 2-1"
model_name = "ResNet101"
entity = "sokrkose"


wandb.init(project = project_name, 
           entity = entity, 
           name = model_name)

wandb.config = {
    "image_size": image_size,
    "optimizer": optimizer,
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size
}

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset='training',
    seed=123,
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size = image_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset='validation',
    seed=123,
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


class_names = train_ds.class_names
print(class_names)

class_names1 = test_ds.class_names
print(class_names1)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


base_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top = False, 
                                                        input_shape = image_size + (3,), 
                                                        weights = 'imagenet')

base_model.trainable = False

myModel = tf.keras.models.Sequential()

myModel.add(tf.keras.layers.experimental.preprocessing.Rescaling((1./255)))
myModel.add(base_model)
myModel.add(tf.keras.layers.GlobalAveragePooling2D())
myModel.add(tf.keras.layers.Dropout(0.3))
myModel.add(tf.keras.layers.Dense(512, activation='relu'))
myModel.add(tf.keras.layers.Dropout(0.3))
myModel.add(tf.keras.layers.Dense(len(class_names), activation='softmax'))

myModel.compile(optimizer = optimizer,
                loss = loss,
                metrics = metrics)

history = myModel.fit(train_ds,
                      validation_data = val_ds,
                      epochs = 1,
                      callbacks = [WandbCallback()])

(eval_loss, eval_accuracy) = myModel.evaluate(test_ds, 
                                              batch_size = batch_size, 
                                              verbose = 1)

print('[INFO] accuracy: {:.2f}%'.format(eval_accuracy * 100)) 
print('[INFO] Loss: {}'.format(eval_loss)) 

myModel.summary()

myModel.save(f'resnet50_700image_15ep.h5')



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize = (8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label = 'Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc = 'lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label = 'Training Loss')
plt.plot(epochs_range, val_loss, label = 'Validation Loss')
plt.legend(loc = 'upper right')
plt.title('Training and Validation Loss')
plt.show()




img = '2016_Copy/test/1/ISIC_0010175.jpg'
PIL.Image.open(img)

img = tf.keras.preprocessing.image.load_img(img, target_size = image_size)
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
predictions = myModel.predict(img_array)
score = tf.nn.softmax(predictions[0])
print(f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score)} percent confidence.")



plt.figure(figsize = (10, 10))
for images, labels in test_ds.take(1):
  for i in range(7):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8")) 
    plt.title(class_names1[labels[i]])
    plt.axis("off")
    

list_predictions = []
list_labels = []

for image, labels in test_ds.as_numpy_iterator():
  for a in range(len(labels)):
    predictions = myModel.predict(image[a].reshape(1, 224, 224, 3))
    list_predictions.append(class_names1[np.argmax(predictions)])
    list_labels.append(class_names1[labels[a]])

print('Accuarcy score: ', accuracy_score(list_labels, list_predictions))
print(classification_report(list_labels, list_predictions))
print(confusion_matrix(list_labels, list_predictions))

preds = list(map(int, list_predictions))
y_true = list(map(int, list_labels))

wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs = None,
                                                    preds = preds, 
                                                    y_true = y_true,
                                                    class_names = class_names1)})

# wandb.log({"image_size": image_size,
#             "optimizer": optimizer,
#             "loss function": loss,
#             "learning_rate": learning_rate,
#             "epochs": epochs,
#             "batch_size": batch_size})

# wandb.log({"roc" : wandb.plot.roc_curve(list_labels,
#                                         predictions,
#                                         labels = class_names1)})
