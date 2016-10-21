from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Activation, Bidirectional
from keras.callbacks import EarlyStopping
import cPickle
import codecs
import utils
import numpy as np
from datetime import datetime
import itertools


time_step = 205
data_dim = 200
num_tag = 15
num_hidden_node = 100
batch_size = 1000


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


def create_data_1(word_file, word_vector_dict):
    input_data = []
    f1 = codecs.open(word_file, 'r', 'utf-8')
    for line1 in f1:
        input = map(int, line1.split())
        input_vector = [word_vector_dict[i] for i in input]
        input_data.append(input_vector)
    input_data = np.asarray(input_data)
    f1.close()
    return input_data


startTime = datetime.now()

print 'Load word vector dict'
with open('word_vector_dict.pkl', 'rb') as input:
    word_vector_dict = cPickle.load(input)
print 'Create data to train'
input_train, output_train = create_data('train-word-id-pad.txt', 'train-tag-id-pad.txt', word_vector_dict)
#input_val, output_val = create_data('testa-word-id-pad.txt', 'testa-tag-id-pad.txt', word_vector_dict)
input_test = create_data_1('test-word-id-pad.txt', word_vector_dict)
print np.shape(input_train), np.shape(output_train), np.shape(input_test)
print 'Create model'
early_stopping = EarlyStopping()
model = Sequential()
model.add(Bidirectional(LSTM(num_hidden_node, return_sequences=True), merge_mode='concat', input_shape=(time_step, data_dim)))
#model.add(LSTM(num_hidden_node, return_sequences=True))
model.add(TimeDistributed(Dense(num_tag)))
model.add(Activation('softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print model.summary()
print np.shape(model.get_weights())

print 'Training'
history = model.fit(input_train, output_train, batch_size=batch_size, nb_epoch=10, validation_split=0.2,
                    callbacks=[])
#score, acc = model.evaluate(input_test, output_test, batch_size=batch_size)
answer = model.predict_classes(input_test, batch_size=batch_size)
#print np.shape(answer)
#test = np.argmax(output_test, axis=2)
utils.predict_to_file('test-predict-id-pad.txt', answer)
#print('Test score:', score)
#print('Test accuracy:', acc)
#acc1 = utils.evaluate(answer, test)

#with open('le.pkl', 'rb') as input:
#    le = cPickle.load(input)
#utils.convert_to_conll_format('testb-predict-id-pad.txt', 'testb-tag.txt', 'testb-word.txt', le)
endTime = datetime.now()
print "Running time: "
print (endTime - startTime)

