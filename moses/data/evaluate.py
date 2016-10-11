import codecs
import os
import itertools


def evaluate_data(filename1, filename2, path):
    f1 = codecs.open(os.path.join(path, filename1), 'r', 'utf-8')
    f2 = codecs.open(os.path.join(path, filename2), 'r', 'utf-8')
    count1 = 0
    count2 = 0
    for line1, line2 in itertools.izip(f1, f2):
        line1 = line1.split()
        line2 = line2.split()
        for word1, word2 in itertools.izip(line1, line2):
            count1 += 1
            if word1 == word2:
                count2 += 1
            else:
                print word1 + ' : ' + word2
    print count2/float(count1)

if __name__ == "__main__":
    evaluate_data('clean.testb.predict', 'clean.testb.tag', 'test')