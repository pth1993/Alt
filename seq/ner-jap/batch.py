import subprocess


dropout_list = []
num_hidden_node_list = []
f = open('input_para.txt', 'r')
num_hidden_node_list = f.readline().strip().split()
dropout_list = f.readline().strip().split()
#print num_hidden_node_list
#print dropout_list
f.close()
for num_hidden_node in num_hidden_node_list:
    for dropout in dropout_list:
        #print num_hidden_node, dropout
        subprocess.call(['python', 'lstm_batch.py', num_hidden_node, dropout])
