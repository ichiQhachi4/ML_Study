
import math
import random
import numpy as np
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
sigmoid_v = np.vectorize(sigmoid)

@staticmethod
def gradient(Y_o, Y_t): # y_o output by nn, Y_t label
    return Y_o * (1 - Y_o) * (Y_t * Y_o)


gradient_v = np.vectorize(gradient)

@staticmethod
def gradient2(output, acc):
    return output * (1 - output) * acc

gradient2_v = np.vectorize(gradient2)

class Neural_Network(object):
    def __init__(self, X, Y):
        # initial parameters
        self.X_train = X
        self.Y_train = Y
        self.inputSize = X.shape[1]
        self.outputSize = Y.shape[1]
        self.times = X.shape[0]       
        
        self.hiddenSize = 20
        self.eta = 0.05

        # initial weight with random variables
        self.V = np.random.randn(self.inputSize, self.hiddenSize)
        self.W = np.random.randn(self.hiddenSize, self.outputSize)
        self.GAMMA = np.zeros((1, self.hiddenSize))
        self.THETA = np.zeros((1, self.outputSize))
        self.B = np.zeros((1, self.hiddenSize))
        self.output = np.zeros((1, self.outputSize))
        self.Y = np.zeros((1, self.outputSize))
        self.X = np.zeros((1, self.inputSize))
        self.g = np.zeros((1, self.outputSize))
        self.e = np.zeros((1, self.hiddenSize))


        
    # calculate output
    def output_ex(self):
        ALPHA = np.dot(self.X, self.V)
        self.B = sigmoid_v(ALPHA - self.GAMMA)
        '''
        for h in range(self.hiddenSize):
            for i in range(self.inputSize):
                self.output_H[h] += self.V[i][h] * X[i]
            self.output_H[h] = sigmoid_v(self.output_H[h] - self.GAMMA[h])
        '''
        BETA = np.dot(self.B, self.W)
        self.Y = sigmoid_v(BETA - self.THETA)

        '''
        for j in range(self.outputSize):
            for h in range(self.hiddenSize):
                self.output[j] += self.W[h][j] * self.output_H[h]
            self.output[j] = sigmoid_v(self.output[j] - self.THETA[j])
        '''
    
    def generate(self, X):
        output_H = np.dot(X, self.V)
        output_H = sigmoid_v(output_H - self.GAMMA)
        output = np.dot(output_H, self.W)
        return sigmoid_v(output - self.THETA)



    # calculate gradient of weight between Hidden and Output
    def g_ex(self, Y):
        # print(self.Y)
        # print(Y)
        
        #self.g = gradient_v(self.Y, Y)
        # the vectorize function can't be applied to two vector!!!
        for j in range(self.outputSize):
            self.g[0][j] = self.Y[0][j] * (1 - self.Y[0][j]) * (Y[j] - self.Y[0][j])
        
    


    # calculate gradient of weight between Input and Hidden
    def e_ex(self):
        # acc = self.W * self.g
        # self.e = gradient2_v(self.output_H, acc)
        
        for h in range(self.hiddenSize):
            acc = 0.0
            for j in range(self.outputSize):
                acc += self.W[h][j] * self.g[0][j]
            self.e[0][h] = self.B[0][h] * (1 - self.B[0][h]) * acc


    def update(self):
        self.THETA = self.THETA - self.eta * self.g
        self.GAMMA = self.GAMMA - self.eta * self.e
        self.W = self.W + self.eta * np.dot(np.transpose(self.B), self.g)
        self.V = self.V + self.eta * np.dot(np.transpose(self.X), self.e)
        '''
        for j in range(self.outputSize):
            self.THETA[j] = self.THETA[j] - self.eta * self.g[j]
            for h in range(self.hiddenSize):
                self.W[h][j] = self.W[h][j] + self.eta * self.g[j] * self.output_H[h]
        for h in range(self.hiddenSize):
            self.GAMMA[h] = self.GAMMA[h] - self.eta * self.e[h]
            for i in range(self.inputSize):
                self.V[i][h] = self.V[i][h] + self.eta * self.e[h] * self.X[i]
        '''

    def train(self, epoches=100):
        for k in range(epoches):
            for i in range(self.times):
                self.X = self.X_train[i].reshape(1,self.inputSize)
                self.output_ex()

                self.g_ex(self.Y_train[i])
                self.e_ex()
                self.update()
                self.output_ex()

            
            

            
    def test(self, X, Y):
        Y_o = self.generate(X).reshape(self.outputSize)


        result = True
        for i in range(self.outputSize):
            if Y[i] != np.round(Y_o[i]):
                return False
        
        return result


