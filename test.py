from __future__ import print_function
import numpy as np

np.random.seed(1337)
from PIL import Image
from os import listdir
import os

import cv2
import numpy as np

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys

sys.argv[0] # prints python_script.py
FileName=sys.argv[1] # prints var1

img_rows, img_cols = 31,100

model = Model.ModelKeras()

model.load_weights('weights.hdf5')

# img = cv2.imread('test.bmp')
img = cv2.imread(FileName)
col_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# col_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
image = cv2.resize(col_img, (100, 31))
img = np.asarray(image)
img = img.reshape(1, img_rows, img_cols,1 )

Class = model.predict(img)
Class = Class.argmax(axis=-1)
Class = Class[0]
Class_Names = {0: 'Bisacodyl', 1: 'Captopril', 2: 'Enalapril',3: 'Famotidine', 4:'Halazone', 5: 'Lisinopril',6: 'Pancreatin', 7: 'Pantoprazole', 8: 'Sennoside', 9: 'Sucralfate'}
print(Class_Names[Class])