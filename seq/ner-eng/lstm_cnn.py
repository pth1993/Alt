from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Activation, Bidirectional, Input, Convolution1D
from keras.callbacks import EarlyStopping
from keras import objectives
import cPickle
import codecs
import utils
import numpy as np
from datetime import datetime
import itertools
import subprocess, shlex
import os
import argparse
from keras import backend as K
from theano import tensor as T
from theano import shared
from theano.tensor import basic as tensor


input_train_word = np.random.rand(100,10,20)
output_train_word = np.random.rand(100,10,2)
input_dev_word = np.random.rand(100,10,20)
output_dev_word = np.random.rand(100,10,2)
input_test_word = np.random.rand(100,10,20)
output_test_word = np.random.rand(100,10,2)

print 'Create model'
early_stopping = EarlyStopping()
model1 = Sequential()
model1.add(Input(shape=(None,10,20)))
model2 = Sequential()
model2.add(Convolution1D(10, 5, border_mode='same', input_dim=10))

model.add(Bidirectional(LSTM(20, return_sequences=True), merge_mode='concat', input_shape=(10, 20)))
model.add(TimeDistributed(Dense(2)))
model.add(Activation('softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])