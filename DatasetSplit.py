# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 13:51:53 2021

@author: Sokratis Koseoglou
"""

import pandas as pd
import os
import shutil

# os.path.abspath(os.getcwd())
# os.chdir('./Desktop/Thesis/Codes/2017_Copy')

train_groundTruth = pd.read_csv('ISIC-2017_Training_Part3_GroundTruth.csv', names = ['image_id', 'melanoma', 'seborrheic_keratosis'])
validation_groundTruth = pd.read_csv('ISIC-2017_Validation_Part3_GroundTruth.csv', names = ['image_id', 'melanoma', 'seborrheic_keratosis'])
test_groundTruth = pd.read_csv('ISIC-2017_Test_v2_Part3_GroundTruth.csv', names = ['image_id', 'melanoma', 'seborrheic_keratosis'])

# print(validation_groundTruth['melanoma'][26] == '1.0')

src1 = 'ISIC-2017_Training_Data'
dst1 = 'train/1'
dst2 = 'train/0'

src2 = 'ISIC-2017_Validation_Data'
dst3 = 'validation/1'
dst4 = 'validation/0'

src3 = 'ISIC-2017_Test_v2_Data'
dst5 = 'test/1'
dst6 = 'test/0'

for i in range(1, len(train_groundTruth)):
    if (train_groundTruth['melanoma'][i] == '1.0'):
        srcNew = os.path.join(src1, train_groundTruth['image_id'][i] + '.jpg')
        shutil.move(srcNew, dst1)
    else:
        srcNew = os.path.join(src1, train_groundTruth['image_id'][i] + '.jpg')
        shutil.move(srcNew, dst2)


for i in range(1, len(validation_groundTruth)):
    if (validation_groundTruth['melanoma'][i] == '1.0'):
        srcNew = os.path.join(src2, validation_groundTruth['image_id'][i] + '.jpg')
        shutil.move(srcNew, dst3)
    else:
        srcNew = os.path.join(src2, validation_groundTruth['image_id'][i] + '.jpg')
        shutil.move(srcNew, dst4)
     
        
for i in range(1, len(test_groundTruth)):
    if (test_groundTruth['melanoma'][i] == '1.0'):
        srcNew = os.path.join(src3, test_groundTruth['image_id'][i] + '.jpg')
        shutil.move(srcNew, dst5)
    else:
        srcNew = os.path.join(src3, test_groundTruth['image_id'][i] + '.jpg')
        shutil.move(srcNew, dst6)
        