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


def generate_sample_word2vec(filename1, filename2):
    f1 = codecs.open(filename1, 'r', 'utf-8')
    f2 = codecs.open(filename2, 'w', 'utf-8')
    count = 0
    for line in f1:
        count += 1
        if count <= 100:
            f2.write(line)
        else:
            break
    f1.close()
    f2.close()



if __name__ == "__main__":
    #read_corpus('corpus-word-reduce-num.txt')
    #read_corpus('corpus-word-id.txt')
    #compare_line('corpus-word.txt', 'corpus-word-id.txt', 'corpus-word-reduce-num.txt')
    #check_word_vector_dict('GoogleNews-vectors-negative300.txt')
    generate_sample_word2vec('GoogleNews-vectors-negative300.txt', 'word2vec_sample.txt')
