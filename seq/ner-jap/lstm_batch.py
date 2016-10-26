from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Activation, Bidirectional
from keras.callbacks import EarlyStopping
import cPickle
import codecs
import utils
import numpy as np
from datetime import datetime
import itertools
import subprocess, shlex
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("num_hidden_node", help="number of hidden node")
parser.add_argument("dropout", help="dropout number")
args = parser.parse_args()
with open('parameter.pkl', 'rb') as input:
    parameter = cPickle.load(input)
time_step = parameter[0]
data_dim = 200
num_tag = parameter[2] + 1
num_hidden_node = int(args.num_hidden_node)
batch_size = 1000
dropout = float(args.dropout)

print 'Time step: ' + str(time_step)
print 'Data dim: ' + str(data_dim)
print 'Num word: ' + str(parameter[1])
print 'Num tag: : ' + str(num_tag-1)
print 'Num hidden node: ' + str(num_hidden_node)
print 'Batch size: ' + str(batch_size)
print 'Dropout: ' + str(dropout)


def create_data(word_file, tag_file, word_vector_dict):
    input_data = []
    output_data = []
    f1 = codecs.open(word_file, 'r', 'utf-8')
    f2 = codecs.open(tag_file, 'r', 'utf-8')
    for line1, line2 in itertools.izip(f1, f2):
        input = map(int, line1.split())
        output = map(int,line2.split())
        input_vector = [word_vector_dict[i] for i in input]
        output_vector = np.eye(num_tag)[output]
        input_data.append(input_vector)
        output_data.append(output_vector)
    input_data = np.asarray(input_data)
    output_data = np.asarray(output_data)
    f1.close()
    f2.close()
    return input_data, output_data


startTime = datetime.now()

print 'Load word vector dict'
with open('word_vector_dict.pkl', 'rb') as input:
    word_vector_dict = cPickle.load(input)

print 'Create data to train'
input_train, output_train = create_data('train-word-id-pad.txt', 'train-tag-id-pad.txt', word_vector_dict)
input_test, output_test = create_data('test-word-id-pad.txt', 'test-tag-id-pad.txt', word_vector_dict)

print np.shape(input_train), np.shape(output_train), np.shape(input_test), np.shape(output_test)

print 'Create model'
early_stopping = EarlyStopping()
model = Sequential()
model.add(Bidirectional(LSTM(num_hidden_node, return_sequences=True, dropout_W=dropout, dropout_U=dropout), merge_mode='concat', input_shape=(time_step, data_dim)))
model.add(TimeDistributed(Dense(num_tag)))
model.add(Activation('softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print model.summary()
print np.shape(model.get_weights())

print 'Training'
history = model.fit(input_train, output_train, batch_size=batch_size, nb_epoch=5, validation_split=0.0,
                    callbacks=[])
weights = model.get_weights()
np.save('model/weight' + str(num_hidden_node) + '_' + str(dropout), weights)
answer = model.predict_classes(input_test, batch_size=batch_size)
utils.predict_to_file('test-predict-id.txt', 'test-tag-id.txt', answer, num_tag-1)
with open('le_word.pkl', 'rb') as input:
    le_word = cPickle.load(input)
with open('le_tag.pkl', 'rb') as input:
    le_tag = cPickle.load(input)
utils.convert_to_conll_format('test-predict-id.txt', 'test-tag-id.txt', 'test-word-id.txt', le_word, le_tag, num_tag-1)
input = open('conll_output.txt')
output = open(os.path.join('evaluate', 'evaluate' + '_' + str(num_hidden_node) + '_' + str(dropout) + '.txt'), 'w')
subprocess.Popen(shlex.split("perl conlleval.pl"), stdin=input, stdout=output)
endTime = datetime.now()
output.write('Running time: ' + str(endTime-startTime) + '\n')
print "Running time: "
print (endTime - startTime)

