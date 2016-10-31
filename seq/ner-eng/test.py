import codecs
import itertools
from gensim import corpora
import cPickle
import numpy as np


def read_corpus(filename):
    f = codecs.open(filename, 'r', 'utf-8')
    corpus = []
    for line in f:
        line = line.split()
        corpus.append(line)
    dictionary = corpora.Dictionary(corpus)
    print dictionary


def compare_line(filename1, filename2, filename3):
    f1 = codecs.open(filename1, 'r', 'utf-8')
    f2 = codecs.open(filename2, 'r', 'utf-8')
    f3 = codecs.open(filename3, 'r', 'utf-8')
    count = 0
    for line1, line2, line3 in itertools.izip(f1, f2, f3):
        count += 1
        line1 = line1.split()
        line2 = line2.split()
        line3 = line3.split()
        if len(line1) == len(line2) == len(line3):
            pass
        else:
            print count


def check_word_vector_dict(filename):
    with open('word_vector_dict.pkl', 'rb') as input:
        word_vector_dict = cPickle.load(input)
    with open('word_dict.pkl', 'rb') as input:
        word_dict = cPickle.load(input)
    print np.shape(word_vector_dict)
    print np.shape(word_dict)
    index = word_dict.index('for')
    print index
    print word_dict[index]
    print word_vector_dict[index]
    """for line in word_dict:
        line = line.split()
        if len(line) > 1:
            print line
    #print word_dict
    index_list = []
    vector_list = []
    f = codecs.open(filename, 'r', 'utf-8')
    for line in f:
        line = line.split()
        try:
            index = word_dict.index(line[0])
            vector = line[1:301]
            vector = [float(i) for i in vector]
            vector_list.append(vector)
            index_list.append(index)
        except:
            pass
    print len(vector_list)
    print len(index_list)"""



if __name__ == "__main__":
    read_corpus('corpus-word-reduce-num.txt')
    read_corpus('corpus-word-id.txt')
    compare_line('corpus-word.txt', 'corpus-word-id.txt', 'corpus-word-reduce-num.txt')
    check_word_vector_dict('GoogleNews-vectors-negative300.txt')
