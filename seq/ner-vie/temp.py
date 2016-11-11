#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs


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
f2.close()"""
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