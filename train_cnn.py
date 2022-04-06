import pandas as pd
import glob
from PIL import Image
import numpy as np
import keras
from keras.utils import to_categorical
import tensorflow as tf
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from time import sleep
import os
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

np.random.seed(42)

tf.config.run_functions_eagerly(True)
from keras import backend as K

# import keras
"""
#paths = ['all_letters', 'letters_vitkin/correct', 'letters_yoni/correct']
paths=['letters_from_all_folders/correct']
X = []
y = []
for i in range(1, 28):
    print(i)
    filelist = []
    for p in paths:
        filelist += glob.glob('{}/{}/*.png'.format(p, i))
    res = np.array([np.array(Image.open(fname)) for fname in filelist])
    res = np.unique(res.reshape(res.shape[0], -1), axis=0).reshape(res.shape[0], 60, 60)
    res_y = [i] * res.shape[0]
    if type(X) is list:
        X = res
        y = res_y
    else:
        X = np.vstack([X, res])
        y = y + res_y

y = to_categorical(y)
y = y[:, 1:]

pd.to_pickle(X,'train_data/X_23042021.pkl')
pd.to_pickle(y,'train_data/y_23042021.pkl')
"""
# # #X = pd.read_pickle('train_data/X.pkl')
# # #y = pd.read_pickle('train_data/y.pkl')
# # #N = 1000
# # #X = X[np.random.choice(X.shape[0], N, replace=False)]
# #
# # X = X.reshape(X.shape[0], 60, 60, 1)
#
# #y = y[np.random.choice(y.shape[0], N, replace=False)]
# print(X.shape)
# print(y.shape)
X = pd.read_pickle('train_data/X_23042021.pkl')
y = pd.read_pickle('train_data/y_23042021.pkl')
X = X.reshape(X.shape[0], 60, 60, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
pd.to_pickle(X_test,'models/X_test.pkl')
pd.to_pickle(y_test,'models/y_test.pkl')
"""
def create_model(neurons=390, activation1='tanh', dropout_rate=0.0, activation2='sigmoid'):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(60, 60, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(27, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, verbose=4)

batch_size = np.arange(1, 128)
nb_epoch = np.arange(1, 100)
dropout_rate = np.arange(0, 0.6, 0.01)

param_distributions = dict(nb_epoch=nb_epoch, dropout_rate=dropout_rate, batch_size=batch_size)

grid = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=2, n_jobs=-1)
grid_result = grid.fit(X_train, y_train, validation_data=(X_test, y_test))

print(grid.best_estimator_.get_params())
"""
# create models
model = Sequential()
activation1 = ['relu', 'sigmoid', 'tanh']
activation2 = ['relu', 'sigmoid']
neurons = np.arange(20, 40)
nb_epoch = np.arange(1, 10)
batch_size = np.arange(20, 100)
param_distributions = dict(batch_size=batch_size, nb_epoch=nb_epoch, activation1=activation1, \
                           activation2=activation2, neurons=neurons)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=1000, n_jobs=-1,
                          random_state=42)
params = []
for i in range(1000):
    d = dict()
    d['activation1'] = np.random.choice(activation1)
    d['activation2'] = np.random.choice(activation2)
    d['neurons'] = np.random.choice(neurons)
    d['nb_epoch'] = np.random.choice(nb_epoch)
    d['batch_size'] = np.random.choice(batch_size)
    params.append(d)
    if i % 50 == 0:
        print(i)
#
# x = grid._get_param_iterator()
# params = list(x)

for i, p in enumerate(params):
    print(p)
    all_files = os.listdir('models/')
    if [f for f in all_files if f.startswith('model{}_'.format(i))]:
        # if os.path.exists('/data/dna/models/model{}_{}.h5'.format(i,str(curr_param_distributions))):
        print("{} exists".format(i))
        continue

    batch_size = p['batch_size']
    nb_epoch = p['nb_epoch']
    activation1 = p['activation1']
    activation2 = p['activation2']
    neurons = p['neurons']
    model=Sequential()
    model.add(Conv2D(64, kernel_size=3, activation=activation1, input_shape=(60, 60, 1)))
    model.add(Conv2D(32, kernel_size=3, activation=activation2))
    model.add(Flatten())
    model.add(Dense(27, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=nb_epoch, batch_size=batch_size)
    #model.fit(train_data, validation_data=valid_data, epochs=nb_epoch , batch_size=batch_size)
    model.save('models/model{}_.h5'.format(i))
    del model
    K.clear_session()


# model = Sequential()
# # add model layers
# model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(60, 60, 1)))
# model.add(Conv2D(32, kernel_size=3, activation='relu'))
# model.add(Flatten())
# model.add(Dense(27, activation='softmax'))
#
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=32)
# # model.save('cnn_model_v3.h5')
# # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
