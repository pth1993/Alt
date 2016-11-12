import subprocess


subprocess.call(['python', 'lstm_batch.py', 'word2vec', '100', '2', '64', 'none', '0.0', '0.5', '50', 'adam',
                 'categorical_crossentropy', '0', '0'])
subprocess.call(['python', 'lstm_batch.py', 'word2vec', '100', '2', '64', 'none', '0.0', '0.5', '50', 'adam',
                 'categorical_crossentropy', '1', '0'])
subprocess.call(['python', 'lstm_batch.py', 'word2vec', '100', '2', '64', 'none', '0.0', '0.5', '50', 'adam',
                 'categorical_crossentropy', '0', '1'])
subprocess.call(['python', 'lstm_batch.py', 'word2vec', '100', '2', '64', 'none', '0.0', '0.5', '50', 'adam',
                 'categorical_crossentropy', '1', '1'])