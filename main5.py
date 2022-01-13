# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 11:49:35 2022

@author: Sokratis Koseoglou
"""

import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.layers import Flatten, Dense, Dropout
# from keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import numpy as np
# import statistics
import wandb
# from wandb import util
from wandb.keras import WandbCallback
from wandb.sdk.data_types import Image
# import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# import PIL
import cv2 as cv
# import glob
import os
from keras import backend as K

# tf.debugging.set_log_device_placement(True)

# os.path.abspath(os.getcwd())
# os.chdir('./Desktop/Thesis/Codes')

project_name = "test 12-1"
model_name = "resnet_v2.ResNet50V2"
entity = "sokrkose"

wandb.init(project = project_name, 
               entity = entity, 
               name = model_name)

dataset = "2017_Copy"

train_dir = os.path.join(dataset, "train")
validation_dir = os.path.join(dataset, "validation")
test_dir = os.path.join(dataset, "test")
all_data_dir = os.path.join(dataset, "all_data")
epochs = 5
img_size = 224
image_size = (img_size, img_size)
batch_size = 32
learning_rate = 0.01
dropout = 0.5
dense = 512
optimizer = tf.keras.optimizers.Adam(learning_rate)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False)
metrics = ['accuracy']

print(f'Train benign length is: {len(os.listdir(os.path.join(train_dir, "0")))}')
print(f'Train malignant length is: {len(os.listdir(os.path.join(train_dir, "1")))}')
print(f'Validation benign length is: {len(os.listdir(os.path.join(validation_dir, "0")))}')
print(f'Validation malignant length is: {len(os.listdir(os.path.join(validation_dir, "1")))}')
print(f'Test benign length is: {len(os.listdir(os.path.join(test_dir, "0")))}')
print(f'Test malignant length is: {len(os.listdir(os.path.join(test_dir, "1")))}')


# # lets augment the malignant class
# dir_path = os.path.join(train_dir, "1")

# for filename in os.listdir(dir_path):
#     image = cv.imread(os.path.join(dir_path, filename))
#     flipped = tf.image.flip_left_right(image) 
#     new = np.array(flipped)
#     name = 'aug_' + filename
#     cv.imwrite(os.path.join(dir_path, name), new)

# print('\n')

# print(f'Train benign length is: {len(os.listdir(os.path.join(train_dir, "0")))}')
# print(f'Train malignant length is: {len(os.listdir(os.path.join(train_dir, "1")))}')
# print(f'Test benign length is: {len(os.listdir(os.path.join(test_dir, "0")))}')
# print(f'Test malignant length is: {len(os.listdir(os.path.join(test_dir, "1")))}')


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    # validation_split = 0.2,
    # subset = 'training',
    # seed = 123,
    color_mode = 'rgb',
    batch_size = batch_size,
    image_size = image_size
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    # validation_split = 0.2,
    # subset = 'validation',
    # seed = 123,
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
class_names1 = test_ds.class_names

base_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top = False, 
                                                        input_shape = image_size + (3,), 
                                                        weights = 'imagenet')

base_model.trainable = False

myModel = tf.keras.models.Sequential()
myModel.add(tf.keras.layers.experimental.preprocessing.Rescaling((1./255)))
myModel.add(base_model)
myModel.add(tf.keras.layers.GlobalAveragePooling2D())
myModel.add(tf.keras.layers.Dropout(dropout))
myModel.add(tf.keras.layers.Dense(dense, activation='relu'))
myModel.add(tf.keras.layers.Dropout(dropout))
myModel.add(tf.keras.layers.Dense(len(class_names), activation='softmax'))

myModel.compile(optimizer = optimizer,
                loss = loss,
                metrics = metrics)

# K.set_value(myModel.optimizer.learning_rate, learning_rate)

print(myModel.optimizer)
print(myModel.optimizer.learning_rate)

history = myModel.fit(train_ds,
                      validation_data = val_ds,
                      epochs = epochs,
                      callbacks = [WandbCallback()])

print('\n')

(eval_loss, eval_accuracy) = myModel.evaluate(test_ds, 
                                              batch_size = batch_size, 
                                              verbose = 1)

print('[INFO] accuracy: {:.2f}%'.format(eval_accuracy * 100)) 
print('[INFO] Loss: {}'.format(eval_loss)) 

myModel.summary()

# myModel.save(f'resnet50_700image_15ep.h5')
    
probas = []
list_predictions = []
list_labels = []

for image, labels in test_ds.as_numpy_iterator():
  for a in range(len(labels)):
    pred = myModel.predict(image[a].reshape(1, img_size, img_size, 3))
    probas.append(pred[0])
    list_predictions.append(class_names1[np.argmax(pred)])
    list_labels.append(class_names1[labels[a]])

cm = confusion_matrix(list_labels, list_predictions)
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
accuracy = accuracy_score(list_labels, list_predictions)
recall = recall_score(list_labels, list_predictions, pos_label = '0')
precision = precision_score(list_labels, list_predictions, pos_label = '0')
f1 = f1_score(list_labels, list_predictions, pos_label = '0')

print(classification_report(list_labels, list_predictions))
print(cm)
print('Accuarcy : ', accuracy)
print('Sensitivity (TPR) : ', sensitivity)
print('Specificity (TNR) : ', specificity)
print('Recall : ', recall)
print('Precision : ', precision)
print('F1-score : ', f1)


# print('image name \t\t\t predicted  \t confidence \t actual class')

# folder = '2016_Copy/test/1'
# actual_class = '1'
# malignant_resutls = np.empty([1, 4])
# for filename in os.listdir(folder):
#     img_name = os.path.join(folder, filename)
#     img = tf.keras.preprocessing.image.load_img(img_name, target_size = image_size)
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)
#     predictions = myModel.predict(img_array)
#     score = tf.nn.softmax(predictions[0])
#     malignant_resutls = np.append(malignant_resutls, [[ img_name, class_names[np.argmax(score)], 100 * np.max(score), actual_class ]], axis = 0)
#     # print(f"{img_name}\t{class_names[np.argmax(score)]}\t{100 * np.max(score)}\t{actual_class}")

# folder = '2016_Copy/test/0'
# actual_class = '0'
# benign_results = np.empty([1, 4])
# for filename in os.listdir(folder):
#     img_name = os.path.join(folder, filename)
#     img = tf.keras.preprocessing.image.load_img(img_name, target_size = image_size)
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)
#     predictions = myModel.predict(img_array)
#     score = tf.nn.softmax(predictions[0])
#     benign_results = np.append(benign_results, [[ img_name, class_names[np.argmax(score)], 100 * np.max(score), actual_class ]], axis = 0)
#     # print(f"{img_name}\t{class_names[np.argmax(score)]}\t{100 * np.max(score)}\t{actual_class}")

# wandb.Image(str(img_name))

# all_results = np.concatenate((malignant_resutls, benign_results), axis = 0)

# fpr, tpr, thresholds = metrics.roc_curve(y_true, preds)
# roc_auc = metrics.auc(fpr, tpr)
# display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
#                                  estimator_name='example estimator')
# display.plot()
# plt.show()

hyperparameters_table = wandb.Table(columns = ["Parameter", "Value"], 
                                    data = [["image_size", str(image_size)], 
                                            ["optimizer", str(optimizer)],
                                            ["loss function", str(loss)],
                                            ["learning_rate", str(learning_rate)],
                                            ["epochs", str(epochs)],
                                            ["batch_size", str(batch_size)]])

metrics_table = wandb.Table(columns = ["Metric", "Value"], 
                            data = [["Accuracy", str(accuracy)], 
                                    ["Sensitivity (TPR)", str(sensitivity)],
                                    ["Specificity (TNR)", str(specificity)],
                                    ["Recall", str(recall)],
                                    ["Precision", str(precision)],
                                    ["F1-score", str(f1)]])

# image_results = wandb.Table(columns = ["image", "predicted class", "confidence of prediction", "actual class"],
#                             data = all_results)

wandb.log({"Hyperparameters Table": hyperparameters_table})
wandb.log({"Metrcis Table": metrics_table})
# wandb.log({"Image Results Table": image_results})

predictions = list(map(int, list_predictions))
labels = list(map(int, list_labels))
wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs = None,
                                                    preds = predictions, 
                                                    y_true = labels,
                                                    class_names = class_names1)})

probas_array = np.array(probas)
y_true_array = np.array(list_labels)
wandb.log({"roc" : wandb.plot.roc_curve(y_true = y_true_array,
                                        y_probas = probas_array,
                                        labels = class_names1)})
