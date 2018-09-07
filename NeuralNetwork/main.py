import pandas as pd
import sys
from BPNN import Neural_Network
import numpy as np

def zero():
    return [0,0,0]

def one():
    return [0,0,1]

def two():
    return [0,1,0]

def three():
    return [0,1,1]

def four():
    return [1,0,0]

def five():
    return [1,0,1]

switcher = {
    0: zero,
    1: one,
    2: two,
    3: three,
    4: four,
    5: five
}

train =pd.read_excel("./dataset/balance_uni_train.xls", )
X = train.loc[:, ['a','b','c']]
Y = train.loc[:,['y']]

X = np.array(X)
Y_o = np.array(Y)
Y = []
for i in range(300):
    
    Y.append(switcher[Y_o[i][0]]())

nn = Neural_Network()
print(nn.eta)
for j in range(200):
    for i in range(300):
        nn.run(X[i], Y[i])

test =pd.read_excel("./dataset/balance_uni_test.xls", )
X_test = test.loc[:, ['a','b','c']]
Y_test = test.loc[:,['y']]
X_test = np.array(X_test)
Y_t = np.array(Y_test)
Y_test = []
for i in range(325):
    Y_test.append(switcher[Y_t[i][0]]())

success = 0
for k in range(300):    
    if nn.test(X[k], Y[k]):
        success += 1
print(success)
print(success / 300)