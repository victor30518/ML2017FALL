import pandas as pd
import numpy as np
import os, sys
import keras
import keras.backend as K
from keras.layers import Input, Embedding, Flatten, Lambda, Dense, Dropout
from keras.layers.merge import add, dot, Concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

# Load Data
users_filename = sys.argv[4]
test_filname = sys.argv[1]
movies_filname = sys.argv[3]

users = pd.read_csv(users_filename, engine='python', sep='::')#.set_index('UserID')
movies = pd.read_csv(movies_filname, engine='python',sep='::')#.set_index('movieID')
movies['Genres'] = movies.Genres.str.split('|')
pd_test = pd.read_csv(test_filname, engine='python', sep=',')

x_test = pd_test[['UserID', 'MovieID']].values

# users.Age = users.Age.astype('category')
# users.Gender = users.Gender.astype('category')
# users.Occupation = users.Occupation.astype('category')
# ratings.MovieID = ratings.MovieID.astype('category')
# ratings.UserID = ratings.UserID.astype('category')

# Load Model & Generate Prediction
model = load_model("MF_Model.h5")
predict = model.predict([x_test[:,0], x_test[:,1]])
with open(sys.argv[2],'w') as f:
    f.write('TestDataID,Rating')
    f.write('\n')
    for id, i in enumerate(predict):
        f.write(str(id+1)+',')
        f.write(str(i[0]))
        f.write('\n')

