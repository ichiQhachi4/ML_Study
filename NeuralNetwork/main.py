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

train =pd.read_csv("./dataset/irisTrainData.txt" )
train = np.array(train)
np.random.shuffle(train)
print(train)
X = train[:, 0:4]
Y = train[:,4:7]
print(X)
print(Y)

X = np.array(X)
Y = np.array(Y)



nn = Neural_Network()
print(nn.eta)
for j in range(400):
    for i in range(120):
        nn.run(X[i], Y[i])

test =pd.read_csv("./dataset/irisTrainData.txt", )
X_test = test.loc[:, ['a','b','c','d']]
Y_test = test.loc[:,['x','y','z']]
X_test = np.array(X_test)
Y_test = np.array(Y_test)



success = 0
for k in range(120):    
    if nn.test(X[k], Y[k]):
        success += 1
print(success)
print(success / 120)