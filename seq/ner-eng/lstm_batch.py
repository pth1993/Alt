from keras.models import Sequential
from keras.regularizers import l1, l2
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
from keras import backend as K
from theano import tensor as T
from theano import shared
from theano.tensor import basic as tensor


parser = argparse.ArgumentParser()
parser.add_argument("word_embedding", help="word embedding type: word2vec, glove or senna")
parser.add_argument("num_epoch", help="number of epoch")
parser.add_argument("num_hidden_node", help="number of hidden node")
parser.add_argument("regularization_type", help="regularization type: None or L1, L2")
parser.add_argument("regularization_number", help="regularization number")
parser.add_argument("dropout", help="dropout number: between 0 and 1")
parser.add_argument("optimizer", help="optimizer algorithm")
parser.add_argument("loss", help="loss function")
args = parser.parse_args()
with open('parameter.pkl', 'rb') as input:
    parameter = cPickle.load(input)
loss = args.loss
optimizer = args.optimizer
word_embedding_name = args.word_embedding
nb_epoch = int(args.num_epoch)
regularization_type = args.regularization_type
regularization_number = float(args.regularization_number)
if regularization_type == 'None':
    regularization = None
elif regularization_type == 'L1':
    regularization = l1(regularization_number)
elif regularization_type == 'L2':
    regularization = l2(regularization_number)
time_step = parameter[0]
data_dim = 300
num_tag = parameter[2]
num_hidden_node = int(args.num_hidden_node)
batch_size = 500
dropout = float(args.dropout)
_EPSILON = 10e-8

print 'Time step: ' + str(time_step)
print 'Data dim: ' + str(data_dim)
print 'Num word: ' + str(parameter[1])
print 'Num tag: : ' + str(num_tag)
print 'Num hidden node: ' + str(num_hidden_node)
print 'Batch size: ' + str(batch_size)
print 'Dropout: ' + str(dropout)


def categorical_crossentropy_new(y_true, y_pred):
    '''Expects a binary class matrix instead of a vector of scalar classes.
    '''
    output = y_pred
    target = y_true
    from_logits = False
    if from_logits:
        output = T.nnet.softmax(output)
    else:
        # scale preds so that the class probas of each sample sum to 1
        output /= output.sum(axis=-1, keepdims=True)
    # avoid numerical instability with _EPSILON clipping
    output = T.clip(output, _EPSILON, 1.0 - _EPSILON)
    coding_dist = output
    true_dist = target
    bias = shared(np.array([50,50,50,50,50,50,50,1,1]))
    return -tensor.sum(true_dist * tensor.log(coding_dist) * bias,
                       axis=coding_dist.ndim - 1)


def create_data(word_file, tag_file, word_vector_dict):
    input_data = []
    output_data = []
    f1 = codecs.open(word_file, 'r', 'utf-8')
    f2 = codecs.open(tag_file, 'r', 'utf-8')
    for line1, line2 in itertools.izip(f1, f2):
        input = map(int, line1.split())
        output = map(int,line2.split())
        input_vector = [word_vector_dict[i] for i in input]
        output_vector = np.eye(num_tag+1)[output]
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
input_dev, output_dev = create_data('dev-word-id-pad.txt', 'dev-tag-id-pad.txt', word_vector_dict)
input_test, output_test = create_data('test-word-id-pad.txt', 'test-tag-id-pad.txt', word_vector_dict)
#input_test = input_train
#output_test = output_train

print np.shape(input_train), np.shape(output_train), np.shape(input_dev), np.shape(output_dev),\
    np.shape(input_test), np.shape(output_test)

print 'Create model'
early_stopping = EarlyStopping()
model = Sequential()
model.add(Bidirectional(LSTM(num_hidden_node, return_sequences=True, dropout_W=dropout, dropout_U=dropout,
                             W_regularizer=regularization, U_regularizer=regularization), merge_mode='concat',
                        input_shape=(time_step, data_dim)))
model.add(TimeDistributed(Dense(num_tag+1, W_regularizer=regularization)))
model.add(Activation('softmax'))
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])
print model.summary()
print np.shape(model.get_weights())

print 'Training'
history = model.fit(input_train, output_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(input_dev, output_dev), callbacks=[])
#history = model.fit(input_train, output_train, batch_size=batch_size, nb_epoch=10)
weights = model.get_weights()
np.save('model/weight' + '_' + str(num_hidden_node) + '_' + str(dropout), weights)
answer = model.predict_classes(input_test, batch_size=batch_size)
utils.predict_to_file('test-predict-id.txt', 'test-tag-id.txt', answer)
#utils.predict_to_file('test-predict-id.txt', 'train-tag-id.txt', answer)
with open('le_word.pkl', 'rb') as input:
    le_word = cPickle.load(input)
with open('le_tag.pkl', 'rb') as input:
    le_tag = cPickle.load(input)
utils.convert_to_conll_format('test-predict-id.txt', 'test-tag-id.txt', 'test-word-id.txt', le_word, le_tag, num_tag)
#utils.convert_to_conll_format('test-predict-id.txt', 'train-tag-id.txt', 'train-word-id.txt', le_word, le_tag, num_tag)
input = open('conll_output.txt')
output = open(os.path.join('evaluate', 'evaluate' + '_' + word_embedding_name + 'num_epoch_' + str(nb_epoch) + '_' +
                           'num_hidden_node_' + str(num_hidden_node) + '_' + 'regularization_' + regularization_type +
                           '_' + str(regularization_number) + '_' + 'dropout_' + str(dropout) + '_' + optimizer + '_' +
                           loss + '.txt'), 'w')
subprocess.Popen(shlex.split("perl conlleval.pl"), stdin=input, stdout=output)
endTime = datetime.now()
output.write('Running time: ' + str(endTime-startTime) + '\n')
print "Running time: "
print (endTime - startTime)