#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import numpy as np
import itertools


"""f1 = codecs.open('test.txt', 'r', 'utf-8')
f2 = codecs.open('test_new.txt', 'w', 'utf-8')
for line in f1:
    if line.startswith(u'\ufeff<title>') or line.startswith(u'<title>') or line.startswith(u'<editor>') or line.startswith(u'-DOCSTART-') or line.startswith(u'<s>') or line == u'\r\n':
        pass
    elif line.startswith(u'</s>'):
        f2.write(u'\n')
    else:
        f2.write(line.strip() + u'\n')
f1.close()
f2.close()
f1 = codecs.open('train.txt', 'r', 'utf-8')
f2 = codecs.open('train_new.txt', 'w', 'utf-8')
for line in f1:
    if line.startswith(u'\ufeff<title>') or line.startswith(u'<title>') or line.startswith(u'<editor>') or line.startswith(u'-DOCSTART-') or line.startswith(u'<s>') or line == u'\r\n':
        pass
    elif line.startswith(u'</s>'):
        f2.write(u'\n')
    else:
        f2.write(line.strip() + u'\n')
f1.close()
f2.close()
f1 = codecs.open('train_new.txt', 'r', 'utf-8')
word = []
pos = []
chunk = []
tag = []
count = 0
for line in f1:
    count += 1
    #print count
    if line != u'\n':
        line = line.strip().split()
        word.append(line[0])
        pos.append(line[1])
        chunk.append(line[2])
        tag.append(line[3])
        if line[2] in [u'B-ORG', u'I-LOC', u'B-PER', u'I-PER', u')', u'3', u'4']:
            #print count
            print '\t'.join(line)
print set(pos)
print set(chunk)
print set(tag)
f1 = codecs.open('corpus-word-reduce-num.txt', 'r', 'utf-8')
len_word = []
count = 0
list_char = []
for line in f1:
    line = line.split()
    for word in line:
        list_char += list(word)
        len_word.append(len(word))
f1.close()
set_char = set(list_char)
print len(set_char)
#for char in set_char:
#    print char
print np.bincount(len_word)
#print len_word"""

def load_to_matrix(word_file, tag_file, pos_file, chunk_file, case_file):
    f1 = codecs.open(word_file, 'r', 'utf-8')
    f2 = codecs.open(tag_file, 'r', 'utf-8')
    f3 = codecs.open(pos_file, 'r', 'utf-8')
    f4 = codecs.open(chunk_file, 'r', 'utf-8')
    f5 = codecs.open(case_file, 'r', 'utf-8')
    word_matrix = []
    tag_matrix = []
    pos_matrix = []
    chunk_matrix =[]
    case_matrix = []
    for line1, line2, line3, line4, line5 in itertools.izip(f1, f2, f3, f4, f5):
        input_word = map(int, line1.split())
        input_pos = map(int, line3.split())
        input_chunk = map(int, line4.split())
        input_case = map(int, line5.split())
        output = map(int, line2.split())
        word_matrix.append(input_word)
        tag_matrix.append(output)
        pos_matrix.append(input_pos)
        chunk_matrix.append(input_chunk)
        case_matrix.append(input_case)
    word_matrix = np.asarray(word_matrix)[0:4]
    tag_matrix = np.asarray(tag_matrix)[0:4]
    pos_matrix = np.asarray(pos_matrix)[0:4]
    chunk_matrix = np.asarray(chunk_matrix)[0:4]
    case_matrix = np.asarray(case_matrix)[0:4]
    f1.close()
    f2.close()
    f3.close()
    f4.close()
    f5.close()
    return word_matrix, tag_matrix, pos_matrix, chunk_matrix, case_matrix

#word_matrix, tag_matrix, pos_matrix, chunk_matrix, case_matrix = load_to_matrix('dev-word-id-pad.txt', 'dev-tag-id-pad.txt', 'dev-pos-id-pad.txt', 'dev-chunk-id-pad.txt', 'dev-case-id-pad.txt')

def generate_data(word_matrix, tag_matrix, pos_matrix, chunk_matrix, case_matrix, batch):
    index = 0
    while(1):
        p = np.random.permutation(len(word_matrix))
        word_matrix_shuffle = word_matrix
        tag_matrix_shuffle = tag_matrix
        pos_matrix_shuffle = pos_matrix
        chunk_matrix_shuffle = chunk_matrix
        case_matrix_shuffle = case_matrix
        print index
        print p
        input_data = []
        output_data = []
        try:
            for i in xrange(batch):
                #print p
                #print index
                input_word = word_matrix_shuffle[index+i]
                input_pos = pos_matrix_shuffle[index+i]
                input_chunk = chunk_matrix_shuffle[index+i]
                input_case = case_matrix_shuffle[index+i]
                output = tag_matrix_shuffle[index+i]
                input_data.append(input_word)
                output_data.append(output)
            index += batch
        except IndexError:
            index = 0
        input_data = np.asarray(input_data)
        output_data = np.asarray(output_data)
        #print input_data
        yield input_data, output_data


word_matrix, tag_matrix, pos_matrix, chunk_matrix, case_matrix = load_to_matrix('dev-word-id-pad.txt', 'dev-tag-id-pad.txt', 'dev-pos-id-pad.txt', 'dev-chunk-id-pad.txt', 'dev-case-id-pad.txt')
f = generate_data(word_matrix, tag_matrix, pos_matrix, chunk_matrix, case_matrix, 2)

for i in range(5):
    f.next()