import codecs
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cPickle
from datetime import datetime
import itertools
import math
import argparse


vector_length = 200


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
    if word_name == 'word':
        with open('word_dict.pkl', 'wb') as output:
            cPickle.dump(word_dict, output, cPickle.HIGHEST_PROTOCOL)
        with open('le_word.pkl', 'wb') as output:
            cPickle.dump(le, output, cPickle.HIGHEST_PROTOCOL)
    elif word_name == 'tag':
        with open('tag_dict.pkl', 'wb') as output:
            cPickle.dump(word_dict, output, cPickle.HIGHEST_PROTOCOL)
        with open('le_tag.pkl', 'wb') as output:
            cPickle.dump(le, output, cPickle.HIGHEST_PROTOCOL)
    return word_dict


def cut_data(filename1, filename2, max_len, word_name, num_word, num_tag):
    f1 = codecs.open(filename1, 'r', 'utf-8')
    f2 = codecs.open(filename2, 'w', 'utf-8')
    if word_name == 'word':
        temp = ((unicode(num_word) + u' ') * max_len)[0:-1]
    elif word_name == 'tag':
        temp = ((unicode(num_tag) + u' ') * max_len)[0:-1]
    for line in f1:
        line = line.split()
        num_bulk = len(line)/max_len+1
        if word_name == 'word':
            for i in range(max_len-len(line)%max_len):
                line.append(unicode(num_word))
        elif word_name == 'tag':
            for i in range(max_len-len(line)%max_len):
                line.append(unicode(num_tag))
        #print line
        for i in range(num_bulk):
            #print line
            #print line[(i*max_len):((i+1)*max_len)]
            new_line = ' '.join(line[(i*max_len):((i+1)*max_len)])
            if new_line != temp:
                f2.write(new_line+'\n')
    f1.close()
    f2.close()


def load_data(filename):
    f = codecs.open(filename, 'r', 'utf-8')
    data = []
    for line in f:
        line = map(int,line.split())
        data.append(line)
    return data


def convert_to_onehot(filename, data):
    f = codecs.open(filename, 'w', 'utf-8')
    for line in data:
        print line
        onehot = np.eye(num_word+1)[line]


def load_word2vec(filename):
    f = codecs.open(filename, 'r', 'utf-8', 'ignore')
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
                line_new.append('<number>')
            else:
                line_new.append(word)
        line_new = ' '.join(line_new)
        f2.write(line_new+'\n')
    f1.close()
    f2.close()


def create_word_vector_dict(word_dict, filename):
    vector_list = []
    index_list = []
    word_vector_dict = []
    f = codecs.open(filename, 'r', 'utf-8', 'ignore')
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
    vector_list = [x for (y, x) in sorted(zip(index_list, vector_list))]
    for i in range(len(word_dict)):
        if i in index_list:
            word_vector_dict.append(vector_list[index_list.index(i)])
        else:
            word_vector_dict.append(np.random.uniform(-math.sqrt(3/float(vector_length)), math.sqrt(3/float(vector_length)),
                                                      size=vector_length).tolist())
    word_vector_dict.append(np.random.uniform(-math.sqrt(3 / float(vector_length)), math.sqrt(3 / float(vector_length)),
                                              size=vector_length).tolist())
    with open('word_vector_dict.pkl', 'wb') as output:
        cPickle.dump(word_vector_dict, output, cPickle.HIGHEST_PROTOCOL)
    return word_vector_dict


def split_data(filename_corpus, filename_train, filename_test, num_sent):
    num_sent_train = int(0.8*num_sent)
    f1 = codecs.open(filename_corpus, 'r', 'utf-8')
    f2 = codecs.open(filename_train, 'w', 'utf-8')
    f4 = codecs.open(filename_test, 'w', 'utf-8')
    count = 0
    for line in f1:
        count += 1
        if count <= num_sent_train:
            f2.write(line)
        else:
            f4.write(line)
    f1.close()
    f2.close()
    f4.close()


def export_unknown_word(filename_word2vec, filename_unknown_word, word_dict):
    word2vec_list = load_word2vec(filename_word2vec)
    temp = list(set(word_dict) - set(word2vec_list))
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


def evaluate_pos(predict, test):
    count1 = 0
    count2 = 0
    for line1, line2 in itertools.izip(predict, test):
        for word1, word2 in itertools.izip(line1, line2):
            if word2 != num_tag:
                count1 += 1
                if word2 == word1:
                    count2 += 1
    acc = count2/float(count1)
    print 'Accuracy: ' + str(acc)
    return acc


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


def convert_to_alt_format(filename_predict, le):
    predict_list = []
    f1 = codecs.open(filename_predict, 'r', 'utf-8')
    f2 = codecs.open('output.txt', 'w', 'utf-8')
    for line in f1:
        line = map(int, line.split())
        line = le.inverse_transform(line)
        line = map(unicode, line)
        f2.write(' '.join(line) + u'\n')
    f1.close()
    f2.close()


def read_conll_format_train(filename_raw, filename_word, filename_tag):
    f1 = codecs.open(filename_raw, 'r', 'utf-8')
    f2 = codecs.open(filename_word, 'w', 'utf-8')
    f3 = codecs.open(filename_tag, 'w', 'utf-8')
    temp = []
    word_list = []
    tag_list = []
    num_sent = 0
    for line in f1:
        #print line
        if len(line) > 1:
            line = line.split()
            word_list.append(line[0])
            tag_list.append(line[2].strip())
        else:
            temp.append(len(tag_list))
            f2.write(' '.join(word_list) + '\n')
            f3.write(' '.join(tag_list) + '\n')
            word_list = []
            tag_list = []
            num_sent += 1
    f1.close()
    f2.close()
    f3.close()
    print num_sent
    max_len = max(temp)
    print max_len
    return num_sent, max_len


def read_conll_format_test(filename_raw, filename_word, filename_tag):
    f1 = codecs.open(filename_raw, 'r', 'utf-8')
    f2 = codecs.open(filename_word, 'a', 'utf-8')
    f3 = codecs.open(filename_tag, 'a', 'utf-8')
    word_list = []
    tag_list = []
    num_sent = 0
    for line in f1:
        if len(line) > 1:
            line = line.split('\t')
            word_list.append(line[0])
            tag_list.append(line[2].strip())
        else:
            f2.write(' '.join(word_list) + '\n')
            f3.write(' '.join(tag_list) + '\n')
            word_list = []
            tag_list = []
            num_sent += 1
    f1.close()
    f2.close()
    f3.close()
    print num_sent
    return num_sent


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


if __name__ == "__main__":
    startTime = datetime.now()
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", help="corpus name")
    args = parser.parse_args()
    corpus = args.corpus
    parameter = []
    print 'Read corpus'
    num_sent, max_len = read_conll_format_train('data/'+corpus+'.conll', 'corpus-word.txt', 'corpus-tag.txt')
    parameter.append(max_len)
    #num_sent_test = read_conll_format_train('test.conll', 'corpus-word.txt', 'corpus-tag.txt')
    #convert_test_file('hironsan_token.txt', 'corpus-word.txt')
    
    #print 'Reduce number'
    #convert_number_data('corpus-word.txt', 'corpus-word-reduce-num.txt')

    print 'Convert word to id'
    word_dict = convert_word_to_id('corpus-word.txt', 'corpus-word-id.txt', 'word')
    num_word = len(word_dict)
    parameter.append(num_word)
    print 'Convert tag to id'
    tag_dict = convert_word_to_id('corpus-tag.txt', 'corpus-tag-id.txt', 'tag')
    num_tag = len(tag_dict)
    parameter.append(num_tag)
    print 'Create word vector dict'
    create_word_vector_dict(word_dict, 'jawiki_vector.txt')
    print 'Export unknown word'
    export_unknown_word('jawiki_vector.txt', 'unknown_words.txt', word_dict)

    print 'Split data'
    split_data('corpus-word-id.txt', 'train-word-id.txt', 'test-word-id.txt', num_sent)
    split_data('corpus-tag-id.txt', 'train-tag-id.txt', 'test-tag-id.txt', num_sent)

    print 'Padding data'
    cut_data('train-word-id.txt', 'train-word-id-pad.txt', max_len, 'word', num_word, num_tag)
    cut_data('test-word-id.txt', 'test-word-id-pad.txt', max_len, 'word', num_word, num_tag)
    cut_data('train-tag-id.txt', 'train-tag-id-pad.txt', max_len, 'tag', num_word, num_tag)
    cut_data('test-tag-id.txt', 'test-tag-id-pad.txt', max_len, 'tag', num_word, num_tag)
    with open('parameter.pkl', 'wb') as output:
        cPickle.dump(parameter, output, cPickle.HIGHEST_PROTOCOL)
    endTime = datetime.now()
    print "Running time: "
    print (endTime - startTime)
