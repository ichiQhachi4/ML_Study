import pandas as pd
import sys
from BPNN import Neural_Network
import numpy as np

from sklearn.cross_validation import train_test_split


train =pd.read_csv("./dataset/some_datasets/uspst_uni.txt",sep="\t",header=None)
X = np.array(train)
train =pd.read_csv("./dataset/some_datasets/uspst_uni_label.txt",sep="\t",header=None)
Y = np.array(train)
labels = []
for i in range(Y.shape[0]):
    label = list(('0000' + str(bin(int(Y[i]))).replace('0b', ''))[-4:])
    labels.append(list(map(int, label)))
Y = np.array(labels)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

'''
train = fd.generate_sin_data_for_test()
a = np.array(30 * [1])
b = np.array(30 * [0])
X = np.array(train)
Y = np.concatenate((a,b))
Y = Y.reshape(60,1)
'''

nn = Neural_Network(X_train, Y_train)
print(nn.eta)

'''
test =pd.read_csv("./dataset/irisTestData.txt", )
X_test = test.loc[:, ['a','b','c','d']]
Y_test = test.loc[:,['x','y','z']]
X_test = np.array(X_test)
Y_test = np.array(Y_test)
'''

def tests(nn):
    success = 0
    for k in range(300):    
        if nn.test(X_test[k], Y_test[k]):
            success += 1
    print(success)
    print("precision: " + str(success / 300))
j=0
for i in 20*[10]:
    # convergence at 100-200 epoches
    nn.train(i)
    j+=i
    print("epoches: "+ str(j))
    tests(nn)

