from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Activation, GRU
import cPickle
import codecs
import utils
import numpy as np
from datetime import datetime


time_step = 5
data_dim = 300
num_tag = 47
num_hidden_node = 300


def generate_training_data(word_file, tag_file, word_vector_dict, batch):
    while 1:
        f1 = codecs.open(word_file, 'r', 'utf-8')
        f2 = codecs.open(tag_file, 'r', 'utf-8')
        input_batch = []
        output_batch = []
        for i in xrange(batch):
            line1 = f1.readline()
            line2 = f2.readline()
            if line1 == '':
                f1.seek(0)
                line1 = f1.readline()
            if line2 == '':
                f2.seek(0)
                line2 = f2.readline()
            input = map(int, line1.strip().split())
            output = map(int, line2.strip().split())
            input_vector = [word_vector_dict[i] for i in input]
            output_vector = np.eye(num_tag)[output]
            input_batch.append(input_vector)
            output_batch.append(output_vector)
        input_batch = np.asarray(input_batch)
        output_batch = np.asarray(output_batch)
        #print np.shape(input_batch), np.shape(output_batch)
        yield (input_batch, output_batch)
        f1.close()
        f2.close()


startTime = datetime.now()

print 'Load word vector dict'
with open('word_vector_dict.pkl', 'rb') as input:
    word_vector_dict = cPickle.load(input)
print 'Create model'
model = Sequential()
model.add(LSTM(num_hidden_node, return_sequences=True, input_shape=(time_step, data_dim)))
model.add(LSTM(num_hidden_node, return_sequences=True))
model.add(TimeDistributed(Dense(num_tag)))
model.add(Activation('softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print 'Training'
history = model.fit_generator(generate_training_data('train-word-id-pad.txt', 'train-tag-id-pad.txt', word_vector_dict,
                                                     100),validation_data=generate_training_data('testa-word-id-pad.txt', 'testa-tag-id-pad.txt', word_vector_dict,
                                                     100), nb_val_samples=11900, samples_per_epoch=47400, nb_epoch=2)
score, acc = model.evaluate_generator(generate_training_data('testb-word-id-pad.txt', 'testb-tag-id-pad.txt',
                                                             word_vector_dict, 100), val_samples=11000)
answer = model.predict_generator(generate_training_data('testb-word-id-pad.txt', 'testb-tag-id-pad.txt',
                                                             word_vector_dict, 100), val_samples=11000)
#print np.shape(answer)
label = np.argmax(answer, axis=2)
utils.predict_to_file('testb-predict-id-pad.txt', label)
#print np.shape(label)
print('Test score:', score)
print('Test accuracy:', acc)
endTime = datetime.now()
print "Running time: "
print (endTime - startTime)
