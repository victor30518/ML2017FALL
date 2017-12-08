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

def load_data(test_data_path):    
    # 讀入測試資料
    x_test = []
    with open(test_data_path, 'r') as f:
        f.readline()
        for i, line in enumerate(f):
            data = line.split(str(i) + ',')
            # 讀入句子
            x_test.append(data[1].strip('\n'))

    x_test = np.array(x_test)
    
    return x_test

def model_1():
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1, 200, input_length=38, trainable=False))
    model.add(LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(256, dropout=0.2))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model_2():
    model = Sequential()
    model.add(Embedding(len(tokenizer.word_index)+1, 200, input_length=38, trainable=False))
    model.add(Bidirectional(LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 讀入testing_data
x_test = load_data(sys.argv[1])

# 讀入tokenizer
with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

max_review_length = 38
x_test_dictionary=tokenizer.texts_to_sequences(x_test)
x_test =sequence.pad_sequences(x_test_dictionary, maxlen=max_review_length)

model1 = model_1()
model1.load_weights('model1_weight.h5')
predict1 = model1.predict_proba(x_test)

model2 = model_2()
model2.load_weights('model2_weight.h5')
predict2 = model2.predict_proba(x_test)

predict_merge = predict1 + predict2
model_predict = np.argmax(predict_merge, axis=1)

# 產生上傳kaggle的csv檔
import csv
predict = []
for i in range (len(model_predict)):
    predict.append([str(i),model_predict[i]])

filename = sys.argv[2]
text = open(filename, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","label"])
for i in range(len(predict)):
    s.writerow(predict[i]) 
text.close()