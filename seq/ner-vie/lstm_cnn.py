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
from theano import tensor as T
from theano import shared
from theano.tensor import basic as tensor


parser = argparse.ArgumentParser()
parser.add_argument("word_embedding", help="word embedding type: word2vec, glove, senna")
parser.add_argument("num_epoch", help="number of epoch")
parser.add_argument("num_lstm_layer", help="number of lstm layer")
parser.add_argument("num_hidden_node", help="number of hidden node")
parser.add_argument("regularization_type", help="regularization type: none, l1, l2")
parser.add_argument("regularization_number", help="regularization number")
parser.add_argument("dropout", help="dropout number: between 0 and 1")
parser.add_argument("batch_size", help="batch size for training")
parser.add_argument("optimizer", help="optimizer algorithm: sgd, rmsprop, adagrad, adadelta, adam, adamax, nadam")
parser.add_argument("loss", help="loss function: categorical_crossentropy, categorical_crossentropy_bias")
parser.add_argument("pos", help="pos feature: 1 for using and 0 for vice versa")
parser.add_argument("chunk", help="chunk feature: 1 for using and 0 for vice versa")
parser.add_argument("case", help="case feature: 1 for using and 0 for vice versa")
args = parser.parse_args()
with open('parameter.pkl', 'rb') as input:
    parameter = cPickle.load(input)
num_lstm_layer = int(args.num_lstm_layer)
loss = args.loss
optimizer = args.optimizer
word_embedding_name = args.word_embedding
nb_epoch = int(args.num_epoch)
regularization_type = args.regularization_type
regularization_number = float(args.regularization_number)
pos = int(args.pos)
chunk = int(args.chunk)
case = int(args.case)
time_step = parameter[0]
num_tag = parameter[2]
num_pos = parameter[3]
num_chunk = parameter[4]
if word_embedding_name == 'word2vec':
    data_dim = 300
elif word_embedding_name == 'glove':
    data_dim = 300
elif word_embedding_name == 'senna':
    data_dim = 50
if pos:
    data_dim += (num_pos + 1)
if chunk:
    data_dim += (num_chunk + 1)
if case:
    data_dim += 3
num_hidden_node = int(args.num_hidden_node)
batch_size = int(args.batch_size)
dropout = float(args.dropout)
_EPSILON = 10e-8

print 'Word Embedding: ' + word_embedding_name
print 'Num epoch: ' + str(nb_epoch)
print 'Num hidden node: ' + str(num_hidden_node)
print 'Regularization type: ' + regularization_type
print 'Regularization number: ' + str(regularization_number)
print 'Dropout: ' + str(dropout)
print 'Optimizer: ' + optimizer
print 'Batch size: ' + str(batch_size)
print 'Time step: ' + str(time_step)
print 'Data dim: ' + str(data_dim)
print 'Word dict size: ' + str(parameter[1])
print 'POS dict size: ' + str(num_pos)
print 'Chunk dict size: ' + str(num_chunk)
print 'Tag dict size: : ' + str(num_tag)


def categorical_crossentropy_bias(y_true, y_pred):
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
    bias = shared(np.array([10,10,10,10,10,10,10,1,1]))
    return -tensor.sum(true_dist * tensor.log(coding_dist) * bias,
                       axis=coding_dist.ndim - 1)


def create_data_old(word_file, tag_file, word_vector_dict):
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


def gen_pos(tag):
    one_hot = np.zeros(5)
    if tag == 20 or tag == 23:
        one_hot[0] = 1
    elif tag == 13:
        one_hot[1] = 1
    elif tag == 21 or tag == 22:
        one_hot[2] = 1
    elif tag in [36, 37, 38, 39, 40, 41]:
        one_hot[3] = 1
    else:
        one_hot[4] = 1
    return one_hot


def gen_chunk(tag):
    one_hot = np.zeros(5)
    if tag in [3, 12]:
        one_hot[0] = 1
    elif tag in [ 6, 16,  2,  8]:
        one_hot[1] = 1
    elif tag in [4, 13]:
        one_hot[2] = 1
    elif tag == 17:
        one_hot[3] = 1
    else:
        one_hot[4] = 1
    return one_hot


def create_data(word_file, tag_file, pos_file, chunk_file, case_file, word_vector_dict):
    input_data = []
    output_data = []
    f1 = codecs.open(word_file, 'r', 'utf-8')
    f2 = codecs.open(tag_file, 'r', 'utf-8')
    f3 = codecs.open(pos_file, 'r', 'utf-8')
    f4 = codecs.open(chunk_file, 'r', 'utf-8')
    f5 = codecs.open(case_file, 'r', 'utf-8')
    for line1, line2, line3, line4, line5 in itertools.izip(f1, f2, f3, f4, f5):
        input_word = map(int, line1.split())
        input_pos = map(int, line3.split())
        input_chunk = map(int, line4.split())
        input_case = map(int, line5.split())
        output = map(int, line2.split())
        input_vector_word = [word_vector_dict[i] for i in input_word]
        input_vector_pos = np.eye(num_pos + 1)[input_pos]
        #input_vector_pos = [map(int, list(bin(x)[2:].zfill(6))) for x in input_pos]
        #input_vector_pos = [gen_pos(x) for x in input_pos]
        input_vector_chunk = np.eye(num_chunk + 1)[input_chunk]
        #input_vector_chunk = [map(int, list(bin(x)[2:].zfill(5))) for x in input_chunk]
        #input_vector_chunk = [gen_chunk(x) for x in input_chunk]
        input_vector_case = np.eye(3)[input_case]
        output_vector = np.eye(num_tag + 1)[output]
        input_vector = input_vector_word
        if pos:
            input_vector = np.concatenate((input_vector, input_vector_pos), axis=1)
        if chunk:
            input_vector = np.concatenate((input_vector, input_vector_chunk), axis=1)
        if case:
            input_vector = np.concatenate((input_vector, input_vector_case), axis=1)
        input_data.append(input_vector)
        output_data.append(output_vector)
    input_data = np.asarray(input_data)
    output_data = np.asarray(output_data)
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    return input_data, output_data


startTime = datetime.now()

print 'Load word vector dict'
if word_embedding_name == 'word2vec':
    with open('word_vector_dict_word2vec.pkl', 'rb') as input:
        word_vector_dict = cPickle.load(input)
elif word_embedding_name == 'glove':
    with open('word_vector_dict_glove.pkl', 'rb') as input:
        word_vector_dict = cPickle.load(input)
elif word_embedding_name == 'senna':
    with open('word_vector_dict_senna.pkl', 'rb') as input:
        word_vector_dict = cPickle.load(input)

print 'Create data to train'
input_train, output_train = create_data('train-word-id-pad.txt', 'train-tag-id-pad.txt', 'train-pos-id-pad.txt',
                                        'train-chunk-id-pad.txt', 'train-case-id-pad.txt', word_vector_dict)
input_dev, output_dev = create_data('dev-word-id-pad.txt', 'dev-tag-id-pad.txt', 'dev-pos-id-pad.txt',
                                    'dev-chunk-id-pad.txt', 'dev-case-id-pad.txt', word_vector_dict)
input_test, output_test = create_data('test-word-id-pad.txt', 'test-tag-id-pad.txt', 'test-pos-id-pad.txt',
                                      'test-chunk-id-pad.txt', 'test-case-id-pad.txt', word_vector_dict)

print np.shape(input_train), np.shape(output_train), np.shape(input_dev), np.shape(output_dev),\
    np.shape(input_test), np.shape(output_test)

print 'Create model'
early_stopping = EarlyStopping(patience=3)
model = Sequential()
if regularization_type == 'none' and num_lstm_layer == 1:
    model.add(Bidirectional(
        LSTM(num_hidden_node, return_sequences=True, dropout_W=dropout, dropout_U=dropout),
        merge_mode='concat', input_shape=(time_step, data_dim)))
    model.add(TimeDistributed(Dense(num_tag + 1)))
elif regularization_type == 'l1' and num_lstm_layer == 1:
    model.add(Bidirectional(
        LSTM(num_hidden_node, return_sequences=True, dropout_W=dropout, dropout_U=dropout,
             W_regularizer=l1(regularization_number), U_regularizer=l1(regularization_number)),
        merge_mode='concat', input_shape=(time_step, data_dim)))
    model.add(TimeDistributed(Dense(num_tag + 1, W_regularizer=l1(regularization_number))))
elif regularization_type == 'l2' and num_lstm_layer == 1:
    model.add(Bidirectional(
        LSTM(num_hidden_node, return_sequences=True, dropout_W=dropout, dropout_U=dropout,
             W_regularizer=l2(regularization_number), U_regularizer=l2(regularization_number)),
        merge_mode='concat', input_shape=(time_step, data_dim)))
    model.add(TimeDistributed(Dense(num_tag + 1, W_regularizer=l2(regularization_number))))
elif regularization_type == 'none' and num_lstm_layer == 2:
    model.add(Bidirectional(
        LSTM(num_hidden_node, return_sequences=True, dropout_W=dropout, dropout_U=dropout),
        input_shape=(time_step, data_dim)))
    model.add(Bidirectional(
        LSTM(num_hidden_node, return_sequences=True, dropout_W=dropout, dropout_U=dropout),
        merge_mode='concat'))
    model.add(TimeDistributed(Dense(num_tag + 1)))
elif regularization_type == 'l1' and num_lstm_layer == 1:
    model.add(Bidirectional(
        LSTM(num_hidden_node, return_sequences=True, dropout_W=dropout, dropout_U=dropout,
             W_regularizer=l1(regularization_number), U_regularizer=l1(regularization_number)),
        input_shape=(time_step, data_dim)))
    model.add(Bidirectional(
        LSTM(num_hidden_node, return_sequences=True, dropout_W=dropout, dropout_U=dropout,
             W_regularizer=l1(regularization_number), U_regularizer=l1(regularization_number)),
        merge_mode='concat'))
    model.add(TimeDistributed(Dense(num_tag + 1, W_regularizer=l1(regularization_number))))
elif regularization_type == 'l2' and num_lstm_layer == 1:
    model.add(Bidirectional(
        LSTM(num_hidden_node, return_sequences=True, dropout_W=dropout, dropout_U=dropout,
             W_regularizer=l2(regularization_number), U_regularizer=l2(regularization_number)),
        input_shape=(time_step, data_dim)))
    model.add(Bidirectional(
        LSTM(num_hidden_node, return_sequences=True, dropout_W=dropout, dropout_U=dropout,
             W_regularizer=l2(regularization_number), U_regularizer=l2(regularization_number)),
        merge_mode='concat'))
    model.add(TimeDistributed(Dense(num_tag + 1, W_regularizer=l2(regularization_number))))
model.add(Activation('softmax'))
if loss == 'categorical_crossentropy':
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
elif loss == 'categorical_crossentropy_bias':
    model.compile(optimizer=optimizer, loss=categorical_crossentropy_bias, metrics=['accuracy'])
print model.summary()
print np.shape(model.get_weights())

print 'Training'
history = model.fit(input_train, output_train, batch_size=batch_size, nb_epoch=nb_epoch,
                    validation_data=(input_dev, output_dev), callbacks=[early_stopping])
weights = model.get_weights()
#np.save('model/weight' + '_' + str(num_hidden_node) + '_' + str(dropout), weights)
np.save('model/weight' + '_' + word_embedding_name + '_' + 'num_epoch_' + str(nb_epoch) + '_' + 'num_lstm_layer_' +
        str(num_lstm_layer) + '_' 'num_hidden_node_' + str(num_hidden_node) + '_' + 'regularization_' +
        regularization_type + '_' + str(regularization_number) + '_' + 'dropout_' + str(dropout) + '_' + optimizer +
        '_' + loss + '_batch_size_' + str(batch_size) + '_pos_' + str(pos) + '_chunk_' + str(chunk) +
        '_case_' + str(case), weights)
answer = model.predict_classes(input_test, batch_size=batch_size)
utils.predict_to_file('test-predict-id.txt', 'test-tag-id.txt', answer)
with open('le_word.pkl', 'rb') as input:
    le_word = cPickle.load(input)
with open('le_tag.pkl', 'rb') as input:
    le_tag = cPickle.load(input)
utils.convert_to_conll_format('test-predict-id.txt', 'test-tag-id.txt', 'test-word-id.txt', le_word, le_tag, num_tag)
input = open('conll_output.txt')
output = open(os.path.join('evaluate', 'evaluate' + '_' + word_embedding_name + '_' + 'num_epoch_' + str(nb_epoch) +
                           '_' + 'num_lstm_layer_' + str(num_lstm_layer) + '_' 'num_hidden_node_' + str(num_hidden_node)
                           + '_' + 'regularization_' + regularization_type + '_' + str(regularization_number) + '_' +
                           'dropout_' + str(dropout) + '_' + optimizer + '_' + loss + '_batch_size_' + str(batch_size)
                           + '_pos_' + str(pos) + '_chunk_' + str(chunk) + '_case_' + str(case) + '.txt'), 'w')
subprocess.Popen(shlex.split("perl conlleval.pl"), stdin=input, stdout=output)
endTime = datetime.now()
output.write('Running time: ' + str(endTime-startTime) + '\n')
print "Running time: "
print (endTime - startTime)
