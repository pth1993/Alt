import codecs
import os


def process_data(path, filename):
    f1 = codecs.open(os.path.join(path, filename), 'r', 'utf-8')
    f2 = codecs.open(os.path.join(path, 'clean.'+filename), 'w', 'utf-8')
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


if __name__ == "__main__":
    process_data('train', 'train.word')
    process_data('tune', 'tune.word')
    process_data('test', 'testa.word')
    process_data('test', 'testb.word')
