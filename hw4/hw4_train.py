#!/usr/bin/python
from keras.models import Model, Sequential
from keras.layers import Input, Dense
from keras.layers.recurrent import LSTM
from keras.utils import plot_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.layers import Dropout
from keras.preprocessing import sequence
from keras.layers.wrappers import Bidirectional

import tensorflow as tf
import os, sys
import numpy as np
import pickle
# 套件匯入與設定
# GPU設定
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" #1080
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)

def load_data(train_data_path):
    # 讀入訓練資料
    x_train = []
    y_train = []
    with open(train_data_path, 'r') as f:
        for i, line in enumerate(f):
            data = line.split(' +++$+++ ')
            # 讀入標籤
            label = int(data[0])
            if label == 1:
                y_train.append([0.,1.])
            else:
                y_train.append([1.,0.])
            # 讀入句子
            x_train.append(data[1].strip('\n'))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    return x_train,y_train

def build_model_1():
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1, 200, input_length=38, trainable=True))
    model.add(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(256, dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    return model

def build_model_2():
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1, 200, input_length=38, trainable=True))
    model.add(Bidirectional(LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    
    return model

x_train,y_train = load_data(sys.argv[1])

with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

max_review_length = 38
x_train_dictionary=tokenizer.texts_to_sequences(x_train)
x_train =sequence.pad_sequences(x_train_dictionary, maxlen=max_review_length)

model1 = build_model_1()
model2 = build_model_2()

history = model1.fit(x_train, y_train,validation_split=0.1, epochs=2, batch_size=128)
history = model2.fit(x_train, y_train,validation_split=0.1, epochs=2, batch_size=128)

model1.save_weights("model1_weight")
model2.save_weights("model2_weight")