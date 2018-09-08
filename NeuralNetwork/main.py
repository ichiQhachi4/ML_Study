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

train =pd.read_csv("./dataset/irisTrainData.txt")
train = np.array(train)
np.random.shuffle(train)

X = train[:, 0:4]

Y = train[:, 4:7]



nn = Neural_Network(X, Y)
print(nn.eta)
for i in range(50):
    nn.run()

test =pd.read_csv("./dataset/irisTrainData.txt", )
X_test = test.loc[:, ['a','b','c','d']]
Y_test = test.loc[:,['x','y','z']]
X_test = np.array(X_test)
Y_test = np.array(Y_test)

def tests(nn):
    success = 0
    for k in range(20):    
        if nn.test(X[k], Y[k]):
            success += 1
    print(success)
    print(success / 20)

for i in [50, 100, 300, 500, 1000, 3000]:
    for j in range(i):
        nn.run()
    tests(nn)

