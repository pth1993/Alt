from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed
import cPickle
import codecs
import itertools
import numpy as np
from datetime import datetime


startTime = datetime.now()

time_step = 5
data_dim = 300
num_tag = 46
num_hidden_node = 100

with open('word_vector_dict.pkl', 'rb') as input:
    word_vector_dict = cPickle.load(input)

count2 = 0
for line in word_vector_dict:
    if len(line) != 300:
        print len(line)
        count2 += 1
print count2

"""
def generate_training_data(word_file, tag_file, word_vector_dict):
    while 1:
        f1 = codecs.open(word_file, 'r', 'utf-8')
        f2 = codecs.open(tag_file, 'r', 'utf-8')
        for line1, line2 in itertools.izip(f1, f2):
            input = line1.split()
            output = line2.split()

            yield (x, y)
        f1.close()
        f2.close()

model = Sequential()
model.add(LSTM(num_hidden_node, return_sequences=True, input_shape=(time_step, data_dim)))
model.add(TimeDistributed(Dense(num_tag)))
"""
endTime = datetime.now()
print "Running time: "
print (endTime - startTime)