import codecs
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cPickle
from datetime import datetime
import itertools
import math
import argparse


#vector_length = 300


def convert_word_to_id(filename1, filename2, word_name):
    f1 = codecs.open(filename1, 'r', 'utf-8')
    f2 = codecs.open(filename2, 'w', 'utf-8')
    data1 = []
    data2 = []
    for line in f1:
        line = line.split()
        data1 = data1 + line
        data2.append(line)
    le = LabelEncoder()
    le.fit(data1)
    for line in data2:
        line = le.transform(line).tolist()
        f2.write(' '.join(str(x) for x in line) + '\n')
    f1.close()
    f2.close()
    word_dict = list(le.classes_)
    with open(word_name + '_dict.pkl', 'wb') as output:
        cPickle.dump(word_dict, output, cPickle.HIGHEST_PROTOCOL)
    with open('le_' + word_name + '.pkl', 'wb') as output:
        cPickle.dump(le, output, cPickle.HIGHEST_PROTOCOL)
    return word_dict


def cut_data_old(filename1, filename2, max_len, num_word):
    f1 = codecs.open(filename1, 'r', 'utf-8')
    f2 = codecs.open(filename2, 'w', 'utf-8')
    temp = ((unicode(num_word) + u' ') * max_len)[0:-1]
    for line in f1:
        line = line.split()
        num_bulk = len(line)/max_len+1
        for i in range(max_len - len(line) % max_len):
            line.append(unicode(num_word))
        for i in range(num_bulk):
            new_line = ' '.join(line[(i*max_len):((i+1)*max_len)])
            if new_line != temp:
                f2.write(new_line+'\n')
    f1.close()
    f2.close()


def cut_data(filename1, filename2, max_len, num_word):
    f1 = codecs.open(filename1, 'r', 'utf-8')
    f2 = codecs.open(filename2, 'w', 'utf-8')
    for line in f1:
        line = line.split()
        line += [unicode(num_word)] * (max_len - len(line))
        new_line = ' '.join(line)
        f2.write(new_line + '\n')
    f1.close()
    f2.close()


def load_data(filename):
    f = codecs.open(filename, 'r', 'utf-8')
    data = []
    for line in f:
        line = map(int,line.split())
        data.append(line)
    return data


def load_word2vec(filename):
    f = codecs.open('embedding/'+filename, 'r', 'utf-8', 'ignore')
    word2vec_list = []
    f.readline()
    for line in f:
        try:
            word2vec_list.append(line.split()[0])
        except:
            pass
    f.close()
    return word2vec_list


def convert_number_data(filename1, filename2):
    f1 = codecs.open(filename1, 'r', 'utf-8')
    f2 = codecs.open(filename2, 'w', 'utf-8')
    for line in f1:
        line = line.split()
        line_new = []
        for word in line:
            if any(char.isdigit() for char in word):
                line_new.append(u'<number>')
            elif word in [u',', u'<', u'.', u'>', u'/', u'?', u'..', u'...', u'....', u':', u';', u'"', u"'", u'[',
                          u'{', u']', u'}', u'|', u'\\', u'`', u'~', u'!', u'@', u'#', u'$', u'%', u'^', u'&', u'*',
                          u'(', u')', u'-', u'+', u'=']:
                line_new.append(u'<punct>')
            else:
                line_new.append(word)
        line_new = ' '.join(line_new)
        f2.write(line_new+'\n')
    f1.close()
    f2.close()


def create_word_vector_dict(word_dict, filename, embedding, vector_length):
    vector_list = []
    index_list = []
    word_vector_dict = []
    f = codecs.open('embedding/' + filename, 'r', 'utf-8', 'ignore')
    for line in f:
        line = line.split()
        try:
            index = word_dict.index(line[0])
            vector = line[1:(vector_length+1)]
            vector = [float(i) for i in vector]
            vector_list.append(vector)
            index_list.append(index)
        except:
            pass
    for i in range(len(word_dict)):
        if i in index_list:
            word_vector_dict.append(vector_list[index_list.index(i)])
        else:
            word_vector_dict.append(np.random.uniform(-math.sqrt(3/float(vector_length)), math.sqrt(3/float(vector_length)),
                                                      size=vector_length).tolist())
    word_vector_dict.append(np.random.uniform(-math.sqrt(3 / float(vector_length)), math.sqrt(3 / float(vector_length)),
                                              size=vector_length).tolist())
    with open('word_vector_dict_'+embedding+'.pkl', 'wb') as output:
        cPickle.dump(word_vector_dict, output, cPickle.HIGHEST_PROTOCOL)


def create_word_vector_dict_senna(word_dict, filename, embedding, vector_length):
    vector_list = []
    index_list = []
    word_vector_dict = []
    word_dict_senna = []
    for word in word_dict:
        word_dict_senna.append(word.lower())
    f = codecs.open('embedding/' + filename, 'r', 'utf-8', 'ignore')
    for line in f:
        line = line.split()
        index = [i for i, x in enumerate(word_dict_senna) if x == line[0]]
        if len(index) > 0:
            vector = line[1:(vector_length + 1)]
            vector = [float(i) for i in vector]
            for item in index:
                vector_list.append(vector)
            index_list += index
        else:
            pass
    for i in range(len(word_dict)):
        if i in index_list:
            word_vector_dict.append(vector_list[index_list.index(i)])
        else:
            word_vector_dict.append(np.random.uniform(-math.sqrt(3/float(vector_length)), math.sqrt(3/float(vector_length)),
                                                      size=vector_length).tolist())
    word_vector_dict.append(np.random.uniform(-math.sqrt(3 / float(vector_length)), math.sqrt(3 / float(vector_length)),
                                              size=vector_length).tolist())
    with open('word_vector_dict_'+embedding+'.pkl', 'wb') as output:
        cPickle.dump(word_vector_dict, output, cPickle.HIGHEST_PROTOCOL)


def split_data(filename_corpus, filename_train, filename_dev, filename_test):
    num_sent_train = 14861
    num_sent_dev = 2000
    f1 = codecs.open(filename_corpus, 'r', 'utf-8')
    f2 = codecs.open(filename_train, 'w', 'utf-8')
    f3 = codecs.open(filename_dev, 'w', 'utf-8')
    f4 = codecs.open(filename_test, 'w', 'utf-8')
    count = 0
    for line in f1:
        count += 1
        if count <= num_sent_train:
            f2.write(line)
        elif num_sent_train < count <= num_sent_train + num_sent_dev:
            f3.write(line)
        elif num_sent_train + num_sent_dev < count:
            f4.write(line)
    f1.close()
    f2.close()
    f3.close()
    f4.close()


def export_unknown_word(filename_word2vec, filename_unknown_word, word_dict):
    word2vec_list = load_word2vec(filename_word2vec)
    temp = list(set(word_dict) - set(word2vec_list))
    f = codecs.open(filename_unknown_word, 'w', 'utf-8')
    for item in temp:
        f.write(item + '\n')
    f.close()


def export_unknown_word_senna(filename_word2vec, filename_unknown_word, word_dict):
    word_dict_lower = [i.lower() for i in word_dict]
    word2vec_list = load_word2vec(filename_word2vec)
    temp = list(set(word_dict_lower) - set(word2vec_list))
    f = codecs.open(filename_unknown_word, 'w', 'utf-8')
    for item in temp:
        f.write(item + '\n')
    f.close()


def predict_to_file(filename1, filename2, output):
    f1 = codecs.open(filename1, 'w', 'utf-8')
    f2 = codecs.open(filename2, 'r', 'utf-8')
    for line1, line2 in itertools.izip(output, f2):
        num = len(line2.split())
        line1 = line1[0:num]
        for word in line1:
            f1.write(unicode(word) + u' ')
        f1.write(u'\n')


def convert_to_conll_format(filename_predict, filename_test, filename_word, le_word, le_tag, num_tag):
    word_list = []
    predict_list = []
    test_list = []
    f1 = codecs.open(filename_predict, 'r', 'utf-8')
    f2 = codecs.open(filename_test, 'r', 'utf-8')
    f3 = codecs.open(filename_word, 'r', 'utf-8')
    f4 = codecs.open('conll_output.txt', 'w', 'utf-8')
    for line in f1:
        line = map(int, line.split())
        line = [x if x != num_tag else num_tag-1 for x in line]
        #print set(line)
        line = le_tag.inverse_transform(line)
        line = map(unicode, line)
        line = [x if x != u'OTHER' else u'O' for x in line]
        predict_list.append(line)
    for line in f2:
        line = map(int, line.split())
        #line = [x for x in line if x != num_tag]
        line = le_tag.inverse_transform(line)
        line = map(unicode, line)
        line = [x if x != u'OTHER' else u'O' for x in line]
        test_list.append(line)
    for line in f3:
        line = map(int, line.split())
        #line = [x for x in line if x != num_word]
        line = le_word.inverse_transform(line)
        line = map(unicode, line)
        word_list.append(line)
    for line1, line2, line3 in itertools.izip(word_list,predict_list, test_list):
        for word, predict_tag, test_tag in itertools.izip(line1,line2, line3):
            f4.write(word + u' ' + u'NP' + u' ' + predict_tag + u' ' + test_tag + u'\n')
        f4.write(u'\n')
    f1.close()
    f2.close()
    f3.close()
    f4.close()


def convert_test_file(filename1, filename2):
    word_list = []
    f1 = codecs.open(filename1, 'r', 'utf-8')
    f2 = codecs.open(filename2, 'a', 'utf-8')
    for line in f1:
        #print len(line)
        if len(line) > 1:
            word_list.append(line.strip())
        else:
            #print len(word_list)
            f2.write(' '.join(word_list) + '\n')
            word_list = []
    f1.close()
    f2.close()


def count_corpus(filename):
    f = codecs.open(filename, 'r', 'utf-8')
    num_line = 0
    num_word_list =[]
    for line in f:
        line = line.split()
        num_line += 1
        num_word_list.append(len(line))
    return num_line, max(num_word_list)


def read_conll_format(filename1, filename2, filename3, filename4, filename5):
    f1 = codecs.open(filename1, 'r', 'utf-8')
    f2 = codecs.open(filename2, 'w', 'utf-8')
    f3 = codecs.open(filename3, 'w', 'utf-8')
    f4 = codecs.open(filename4, 'w', 'utf-8')
    f5 = codecs.open(filename5, 'w', 'utf-8')
    word_list = []
    chunk_list = []
    pos_list = []
    tag_list = []
    #count = 0
    for line in f1:
        #count += 1
        #print count
        line = line.split()
        if len(line) > 0:
            #if line[3] == 'B-VP':
                #print line
            word_list.append(line[0].lower())
            pos_list.append(line[1])
            chunk_list.append(line[2])
            tag_list.append(line[3])
        else:
            f2.write(' '.join(word_list) + u'\n')
            f3.write(' '.join(pos_list) + u'\n')
            f4.write(' '.join(chunk_list) + u'\n')
            f5.write(' '.join(tag_list) + u'\n')
            word_list = []
            chunk_list = []
            pos_list = []
            tag_list = []
            #count += 1
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()


if __name__ == "__main__":
    startTime = datetime.now()
    parameter = []
    read_conll_format('vlsp_corpus.txt', 'corpus-word.txt', 'corpus-pos.txt', 'corpus-chunk.txt',
                      'corpus-tag.txt')
    print 'Read corpus'
    num_sent, max_len = count_corpus('corpus-tag.txt')
    print num_sent, max_len
    parameter.append(max_len)
    print 'Reduce number and punct'
    convert_number_data('corpus-word.txt', 'corpus-word-reduce-num.txt')
    print 'Convert word to id'
    word_dict = convert_word_to_id('corpus-word-reduce-num.txt', 'corpus-word-id.txt', 'word')
    num_word = len(word_dict)
    parameter.append(num_word)
    print 'Convert tag to id'
    tag_dict = convert_word_to_id('corpus-tag.txt', 'corpus-tag-id.txt', 'tag')
    num_tag = len(tag_dict)
    parameter.append(num_tag)
    print 'Convert pos to id'
    pos_dict = convert_word_to_id('corpus-pos.txt', 'corpus-pos-id.txt', 'pos')
    num_pos = len(pos_dict)
    parameter.append(num_pos)
    print 'Convert chunk to id'
    chunk_dict = convert_word_to_id('corpus-chunk.txt', 'corpus-chunk-id.txt', 'chunk')
    num_chunk = len(chunk_dict)
    parameter.append(num_chunk)
    print 'Create word vector dict'
    print 'word2vec'
    create_word_vector_dict(word_dict, 'word2vec_embedding.txt', 'word2vec', 300)
    #print 'glove'
    #create_word_vector_dict(word_dict, 'glove_embedding.txt', 'glove', 300)
    #print 'senna'
    #create_word_vector_dict_senna(word_dict, 'senna_embedding.txt', 'senna', 50)
    print 'Export unknown word'
    print 'word2vec'
    export_unknown_word('word2vec_embedding.txt', 'unknown_words_word2vec.txt', word_dict)
    #print 'glove'
    #export_unknown_word('glove_embedding.txt', 'unknown_words_glove.txt', word_dict)
    #print 'senna'
    #export_unknown_word_senna('senna_embedding.txt', 'unknown_words_senna.txt', word_dict)
    print 'Split data'
    split_data('corpus-word-id.txt', 'train-word-id.txt', 'dev-word-id.txt', 'test-word-id.txt')
    split_data('corpus-tag-id.txt', 'train-tag-id.txt', 'dev-tag-id.txt', 'test-tag-id.txt')
    split_data('corpus-pos-id.txt', 'train-pos-id.txt', 'dev-pos-id.txt', 'test-pos-id.txt')
    split_data('corpus-chunk-id.txt', 'train-chunk-id.txt', 'dev-chunk-id.txt', 'test-chunk-id.txt')
    print 'Padding data'
    cut_data('train-word-id.txt', 'train-word-id-pad.txt', max_len, num_word)
    cut_data('dev-word-id.txt', 'dev-word-id-pad.txt', max_len, num_word)
    cut_data('test-word-id.txt', 'test-word-id-pad.txt', max_len, num_word)
    cut_data('train-tag-id.txt', 'train-tag-id-pad.txt', max_len, num_tag)
    cut_data('dev-tag-id.txt', 'dev-tag-id-pad.txt', max_len, num_tag)
    cut_data('test-tag-id.txt', 'test-tag-id-pad.txt', max_len, num_tag)
    cut_data('train-pos-id.txt', 'train-pos-id-pad.txt', max_len, num_pos)
    cut_data('dev-pos-id.txt', 'dev-pos-id-pad.txt', max_len, num_pos)
    cut_data('test-pos-id.txt', 'test-pos-id-pad.txt', max_len, num_pos)
    cut_data('train-chunk-id.txt', 'train-chunk-id-pad.txt', max_len, num_chunk)
    cut_data('dev-chunk-id.txt', 'dev-chunk-id-pad.txt', max_len, num_chunk)
    cut_data('test-chunk-id.txt', 'test-chunk-id-pad.txt', max_len, num_chunk)
    with open('parameter.pkl', 'wb') as output:
        cPickle.dump(parameter, output, cPickle.HIGHEST_PROTOCOL)
    endTime = datetime.now()
    print "Running time: "
    print (endTime - startTime)
