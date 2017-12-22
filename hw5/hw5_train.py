import pandas as pd
import numpy as np
import keras
import keras.backend as K
from keras.layers import Input, Embedding, Flatten, Lambda, Dense, Dropout
from keras.layers.merge import add, dot, Concatenate
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model

import os
import tensorflow as tf
# 套件匯入與設定
# GPU設定
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" #1080
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)

def get_model(n_users,n_items, latent_dim=6666):
    user_input = Input(shape=[1])
    item_input = Input(shape=[1])
    
    user_vec = Embedding(n_users, latent_dim, embeddings_initializer="random_normal", embeddings_regularizer=l2(0.00001))(user_input)
    user_vec = Flatten()(user_vec)
    
    item_vec = Embedding(n_items, latent_dim, embeddings_initializer="random_normal", embeddings_regularizer=l2(0.00001))(item_input)
    item_vec = Flatten()(item_vec)
    
    user_bias = Embedding(n_users,1,embeddings_initializer="zeros")(user_input)
    user_bias = Flatten()(user_bias)
    
    item_bias = Embedding(n_items,1,embeddings_initializer="zeros")(item_input)
    item_bias = Flatten()(item_bias)
    
    r_hat = keras.layers.Dot(axes=1)([user_vec, item_vec])
    r_hat = keras.layers.Add()([r_hat,user_bias,item_bias])
    
    model = keras.models.Model([user_input,item_input], r_hat)
    model.compile(loss="mse",optimizer="adam")
    return model

# Load Data
users_filename = "users.csv"
train_filname = "train.csv"
movies_filname = "movies.csv"

users = pd.read_csv(users_filename, engine='python', sep='::')#.set_index('UserID')
ratings = pd.read_csv(train_filname, engine='python', sep=',')
movies = pd.read_csv(movies_filname, engine='python',sep='::')#.set_index('movieID')
movies['Genres'] = movies.Genres.str.split('|')

x_train = ratings[['UserID', 'MovieID']].values
y_train = ratings['Rating'].values

# users.Age = users.Age.astype('category')
# users.Gender = users.Gender.astype('category')
# users.Occupation = users.Occupation.astype('category')
# ratings.MovieID = ratings.MovieID.astype('category')
# ratings.UserID = ratings.UserID.astype('category')

# 洗亂Data
np.random.seed(66)
permutation = np.random.permutation(len(x_train))
x_train, y_train = x_train[permutation], y_train[permutation]

# Training
model = get_model(6040 + 1, 3952 + 1, 64)
model.summary()
model.fit([x_train[:,0], x_train[:,1]], y_train, epochs=23, batch_size=4096)

model.save("MF_Model.h5")

