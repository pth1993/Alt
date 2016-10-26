import subprocess


dropout_list = []
num_hidden_node_list = []
f = open('input_para.txt', 'r')
for line in f:
    line = line.split()
    num_hidden_node_list.append(line[0])
    dropout_list.append(line[1])
f.close()
for num_hidden_node in num_hidden_node_list:
    for dropout in dropout_list:
        subprocess.Popen([num_hidden_node, dropout])