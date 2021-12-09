# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 13:51:53 2021

@author: Sokratis Koseoglou
"""

import pandas as pd
import os
import shutil

# os.path.abspath(os.getcwd())
# os.chdir('./2016')

train_groundTruth = pd.read_csv('ISBI2016_ISIC_Part3_Training_GroundTruth.csv', names = ['image', 'target'])
test_groundTruth = pd.read_csv('ISBI2016_ISIC_Part3_Test_GroundTruth.csv', names = ['image', 'target'])

src1 = 'ISBI2016_ISIC_Part1_Training_Data'
dst1 = '2016_Copy/train/m'
dst2 = '2016_Copy/train/b'

src2 = 'ISBI2016_ISIC_Part1_Test_Data'
dst3 = '2016_Copy/test/m'
dst4 = '2016_Copy/test/b'

for i in range(len(train_groundTruth)):
    if (train_groundTruth['target'][i] == 'malignant'):
        srcNew = os.path.join(src1, train_groundTruth['image'][i] + '.jpg')
        shutil.move(srcNew, dst1)
    else:
        srcNew = os.path.join(src1, train_groundTruth['image'][i] + '.jpg')
        shutil.move(srcNew, dst2)


for i in range(len(test_groundTruth)):
    if (test_groundTruth['target'][i] == 1):
        srcNew = os.path.join(src2, test_groundTruth['image'][i] + '.jpg')
        shutil.move(srcNew, dst3)
    else:
        srcNew = os.path.join(src2, test_groundTruth['image'][i] + '.jpg')
        shutil.move(srcNew, dst4)
        