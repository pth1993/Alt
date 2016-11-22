import subprocess

for i in range(5):
    #word representation
    """if i > 0:
        subprocess.call(['python', 'lstm_batch.py', 'word2vec', '100', '2', '64', 'none', '0.0', '0.5', '50', 'adam',
                         'categorical_crossentropy', '0', '0', '0', str(i)])
        subprocess.call(['python', 'lstm_batch.py', 'random', '100', '2', '64', 'none', '0.0', '0.5', '50', 'adam',
                         'categorical_crossentropy', '0', '0', '0', str(i)])
    #num_bi-lstm
    subprocess.call(['python', 'lstm_batch.py', 'random', '100', '1', '64', 'none', '0.0', '0.5', '50', 'adam',
                     'categorical_crossentropy', '0', '0', '0', str(i)])
    #drop_out
    subprocess.call(['python', 'lstm_batch.py', 'random', '100', '2', '64', 'none', '0.0', '0.0', '50', 'adam',
                     'categorical_crossentropy', '0', '0', '0', str(i)])
    #feature
    subprocess.call(['python', 'lstm_batch.py', 'word2vec', '100', '2', '64', 'none', '0.0', '0.5', '50', 'adam',
                    'categorical_crossentropy', '1', '0', '0', str(i)])
    subprocess.call(['python', 'lstm_batch.py', 'word2vec', '100', '2', '64', 'none', '0.0', '0.5', '50', 'adam',
                     'categorical_crossentropy', '0', '1', '0', str(i)])
    subprocess.call(['python', 'lstm_batch.py', 'word2vec', '100', '2', '64', 'none', '0.0', '0.5', '50', 'adam',
                     'categorical_crossentropy', '0', '0', '1', str(i)])
    subprocess.call(['python', 'lstm_batch.py', 'word2vec', '100', '2', '64', 'none', '0.0', '0.5', '50', 'adam',
                     'categorical_crossentropy', '1', '1', '0', str(i)])
    subprocess.call(['python', 'lstm_batch.py', 'word2vec', '100', '2', '64', 'none', '0.0', '0.5', '50', 'adam',
                     'categorical_crossentropy', '1', '1', '1', str(i)])"""
    #num_bi-lstm                                                                                                   
    subprocess.call(['python', 'lstm_batch.py', 'word2vec', '100', '1', '64', 'none', '0.0', '0.5', '50', 'adam',
                     'categorical_crossentropy', '0', '0', '0', str(i)])
    #drop_out                                                                                                      
    subprocess.call(['python', 'lstm_batch.py', 'word2vec', '100', '2', '64', 'none', '0.0', '0.0', '50', 'adam',
                     'categorical_crossentropy', '0', '0', '0', str(i)])
