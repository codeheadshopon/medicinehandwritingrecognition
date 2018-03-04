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


TrainImg=[]
TrainLabel=[]

TestImg=[]
TestLabel=[]

# Images = []
# for i in range(1,11):
#     C=0
#     for filename in listdir("medicine/"+str(i)+"/"):
#         if(filename.endswith(".bmp")):
#             Images.append("medicine/"+str(i)+"/"+filename)
# for file in Images:
#     img = cv2.imread(file)
#     imag = np.asarray(img)
#     print(imag.shape)
#     col_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     # col_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     image = cv2.resize(col_img, (480, 150))
#
#     cv2.imwrite(file, image)
#     # print(image.shape)
#

for i in range(1,11):
    C=0
    for filename in listdir("medicine/"+str(i)+"/"):
        if C<180:
            if(filename.endswith(".bmp")):
                TrainImg.append("medicine/"+str(i)+"/"+filename)
                TrainLabel.append(i-1)
                # print("medicine/"+str(i)+"/"+filename)
                C+=1
        else:
            if (filename.endswith(".bmp")):
                TestImg.append("medicine/" + str(i) + "/" + filename)
                TestLabel.append(i-1)
                # print("medicine/"+str(i)+"/"+filename)

print(len(TrainLabel))
print(len(TestLabel))

img_rows, img_cols = 150,480

nb_classes = 10
num_classes = 10
batch_size = 32
epochs = 10


x_train = np.array([np.array(Image.open(fname)) for fname in TrainImg])
x_test = np.array([np.array(Image.open(fname)) for fname in TestImg])

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

y_train = np_utils.to_categorical(TrainLabel, nb_classes)
y_test = np_utils.to_categorical(TestLabel, nb_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')
#
# print(y_train.shape[0], 'train samples')
# print(y_test.shape[0], 'test samples')
#
# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
#
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print(y_train.shape[0], 'train samples')
print(y_test.shape[0], 'test samples')


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print(y_train.shape[0], 'train samples')
print(y_test.shape[0], 'test samples')

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])