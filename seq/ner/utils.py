import codecs
from sklearn.preprocessing import LabelEncoder
import numpy as np
import cPickle
from datetime import datetime
import itertools


num_word = 24886
num_tag = 8
vector_length = 300
num_padding = 124


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
    elif word_name == 'tag':
        with open('tag_dict.pkl', 'wb') as output:
            cPickle.dump(word_dict, output, cPickle.HIGHEST_PROTOCOL)
    return word_dict


def cut_data(filename1, filename2, max_len, word_name):
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
    f = codecs.open(filename, 'r', 'utf-8')
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
    f = codecs.open(filename, 'r', 'utf-8')
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
            word_vector_dict.append(np.random.normal(loc=0.0, scale=1.0, size=vector_length).tolist())
    word_vector_dict.append(np.random.normal(loc=0.0, scale=1.0, size=vector_length).tolist())
    with open('word_vector_dict.pkl', 'wb') as output:
        cPickle.dump(word_vector_dict, output, cPickle.HIGHEST_PROTOCOL)
    return word_vector_dict


def split_data(filename_corpus, filename_train, filename_dev, filename_test):
    f1 = codecs.open(filename_corpus, 'r', 'utf-8')
    f2 = codecs.open(filename_train, 'w', 'utf-8')
    f3 = codecs.open(filename_dev, 'w', 'utf-8')
    f4 = codecs.open(filename_test, 'w', 'utf-8')
    count = 0
    for line in f1:
        count += 1
        if count <= 14987:
            f2.write(line)
        elif count <= (14987+3466):
            f3.write(line)
        else:
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


def predict_to_file(filename, output):
    f = codecs.open(filename, 'w', 'utf-8')
    for line in output:
        for word in line:
            f.write(unicode(word) + u' ')
        f.write(u'\n')


def evaluate(predict, test):
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


if __name__ == "__main__":
    startTime = datetime.now()

    """print 'Reduce number'
    convert_number_data('corpus-word.txt', 'corpus-word-reduce-num.txt')
    print 'Convert word to id'
    word_dict = convert_word_to_id('corpus-word-reduce-num.txt', 'corpus-word-id.txt', 'word')
    print 'Convert tag to id'
    tag_dict = convert_word_to_id('corpus-tag.txt', 'corpus-tag-id.txt', 'tag')
    #print 'Create word vector dict'
    #create_word_vector_dict(word_dict, 'GoogleNews-vectors-negative300.txt')

    #print 'Export unknown word'
    #export_unknown_word('GoogleNews-vectors-negative300.txt', 'unknown_words.txt', word_dict)

    print 'Split data'
    split_data('corpus-word-id.txt', 'train-word-id.txt', 'testa-word-id.txt', 'testb-word-id.txt')
    split_data('corpus-tag-id.txt', 'train-tag-id.txt', 'testa-tag-id.txt', 'testb-tag-id.txt')
"""
    print 'Padding data'
    cut_data('train-word-id.txt', 'train-word-id-pad.txt', num_padding, 'word')
    cut_data('testa-word-id.txt', 'testa-word-id-pad.txt', num_padding, 'word')
    cut_data('testb-word-id.txt', 'testb-word-id-pad.txt', num_padding, 'word')
    cut_data('train-tag-id.txt', 'train-tag-id-pad.txt', num_padding, 'tag')
    cut_data('testa-tag-id.txt', 'testa-tag-id-pad.txt', num_padding, 'tag')
    cut_data('testb-tag-id.txt', 'testb-tag-id-pad.txt', num_padding, 'tag')
    endTime = datetime.now()
    print "Running time: "
    print (endTime - startTime)