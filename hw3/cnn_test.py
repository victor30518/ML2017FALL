# coding: utf-8
import  csv, json, keras, sys, numpy, pickle, os
import pandas as pd
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

from numpy import argmax
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta

from keras.utils import np_utils, plot_model

def load_test_data(test_data_path):
    # 讀入測試資料
    x_test = []
    with open(test_data_path, 'r') as f:
        f.readline()
        for i, line in enumerate(f):
            data = line.split(',')

            # 處理pixel
            pixel = data[1].strip('\n').split(' ')
            x_test.append(pixel)

    x_test = np.array(x_test,dtype=float)

    return x_test

def prediction_generator(model,test_data,output_path):
    predict = []
    one_hot_encoding_predict = model.predict(test_data)
    for i in range (len(one_hot_encoding_predict)):
        y = argmax(one_hot_encoding_predict[i])
        predict.append([str(i),y])

    filename = output_path
    text = open(filename, "w+")
    s = csv.writer(text,delimiter=',',lineterminator='\n')
    s.writerow(["id","label"])
    for i in range(len(predict)):
        s.writerow(predict[i]) 
    text.close()

x_test = load_test_data(sys.argv[1])
x_test = np.resize(x_test, (x_test.shape[0], 48, 48, 1))
x_test = x_test/255

model = load_model("cnn_model.h5")
model.summary()
# print(x_test)
# print(model.predict(x_test))
# predict = model.predict(x_test)
prediction_generator(model,x_test,sys.argv[2])