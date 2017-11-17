#!/usr/bin/env python
# -- coding: utf-8 --
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from numpy import argmax
import csv
import sys

def load_data(train_data_path):
    # 讀入訓練資料
    x_train = []
    y_train = []
    with open(train_data_path, 'r') as f:
        f.readline()
        for i, line in enumerate(f):
            data = line.split(',')
            # 以one_hot_encoding的型式讀進y_train
            label = int(data[0])
            one_hot_encoding = [0.,0.,0.,0.,0.,0.,0.]
            one_hot_encoding[label] = 1.
            y_train.append(one_hot_encoding)

            # 處理pixel
            pixel = data[1].strip('\n').split(' ')
            x_train.append(pixel)

    x_train = np.array(x_train,dtype=float)
    y_train = np.array(y_train)

    return x_train,y_train

def build_model():

    '''
    #先定義好框架
    #第一步從input吃起
    '''
    input_img = Input(shape=(48, 48, 1))
    '''
    先來看一下keras document 的Conv2D
    keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), 
        padding='valid', data_format=None, dilation_rate=(1, 1),
        activation=None, use_bias=True, kernel_initializer='glorot_uniform',
        bias_initializer='zeros', kernel_regularizer=None,
        bias_regularizer=None, activity_regularizer=None,
        kernel_constraint=None, bias_constraint=None)
    '''

    block1 = Conv2D(64, (5, 5), padding='valid', activation='relu',kernel_initializer='glorot_normal')(input_img)
    block1 = ZeroPadding2D(padding=(2, 2), data_format='channels_last')(block1)
    block1 = BatchNormalization()(block1)
    block1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(block1)
    # block1 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block1)

    block2 = Conv2D(128, (3, 3), activation='relu',kernel_initializer='glorot_normal')(block1)
    block2 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block2)
    block2 = BatchNormalization()(block2)
    block2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(block2)

    block3 = Conv2D(128, (3, 3), activation='relu',kernel_initializer='glorot_normal')(block2)
    block3 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block3)
    block3 = BatchNormalization()(block3)
    block3 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(block3)

    # block4 = Conv2D(256, (3, 3), activation='relu',kernel_initializer='glorot_normal')(block3)
    # block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)
    # block4 = BatchNormalization()(block4)
    # block4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1))(block4)

    block4 = Conv2D(512, (3, 3), activation='relu',kernel_initializer='glorot_normal')(block3)
    block4 = ZeroPadding2D(padding=(1, 1), data_format='channels_last')(block4)
    block4 = BatchNormalization()(block4)
    # block4 = MaxPooling2D(pool_size=(2, 2), strides=(1, 1))(block4)
    block4 = AveragePooling2D(pool_size=(3, 3), strides=(2, 2))(block4)
    block4 = Flatten()(block4)

    fc1 = Dense(1024, activation='relu',kernel_initializer='glorot_normal')(block4)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(0.5)(fc1)

    fc2 = Dense(512, activation='relu',kernel_initializer='glorot_normal')(fc1)
    fc2 = BatchNormalization()(fc2)
    fc2 = Dropout(0.5)(fc2)

    predict = Dense(7,kernel_initializer='glorot_normal')(fc2)
    predict = Activation('softmax')(predict)
    model = Model(inputs=input_img, outputs=predict)

    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam(lr=1e-3)
    # opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

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

# This code is using tensorflow backend
#!/usr/bin/env python
# -- coding: utf-8 --
# from cnn_simple import 
# from utils import 
import os
import numpy as np
import argparse
import time
import tensorflow as tf

# os.system('echo $CUDA_VISIBLE_DEVICES')

# # GPU設定
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" #1080
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)

def main():
    # ImageDataGenerator
    imageDataGenerator = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.3,
        height_shift_range=0.3,
        zoom_range=[0.7, 1.3],
        shear_range=0.3,
        horizontal_flip=True)

    # load data
    x_train,y_train = load_data(sys.argv[1])

    # normalize
    x_train = x_train/255

    # split validation
    validation_size = len(x_train)//10
    x_validation = x_train[0:validation_size]
    x_train = x_train[validation_size:]
    y_validation = y_train[0:validation_size]
    y_train = y_train[validation_size:]

    # resize
    x_train = np.resize(x_train, (x_train.shape[0], 48, 48, 1))
    x_validation = np.resize(x_validation, (x_validation.shape[0], 48, 48, 1))  

    # build model & training
    model = build_model()
    # earlyStopping = EarlyStopping(monitor='val_acc', patience=50, verbose=0, mode='auto')
    # modelCheckpoint_filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
    # modelCheckpoint = ModelCheckpoint(modelCheckpoint_filepath, monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    # model.fit(x_train, y_train, epochs=5000, batch_size=200, validation_split=0.1, callbacks=[modelCheckpoint])
    model.fit_generator(imageDataGenerator.flow(x_train, y_train, batch_size=128), steps_per_epoch=2*len(x_train)//128,epochs=600,validation_data=(x_validation, y_validation))

    model.save("cnn_model.h5")
    
if __name__=='__main__':
    main()