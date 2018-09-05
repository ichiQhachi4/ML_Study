import math
import random
import numpy as np
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
sigmoid_v = np.vectorize(sigmoid)

class Neural_Netword(object):
    def __init__(self):
        # initial parameters
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3
        self.eta = 0.05

        # initial weight with random variables
        self.V = np.random.randn(self.inputSize, self.hiddenSize)
        self.W = np.random.randn(self.hiddenSize, self.outputSize)
        self.GAMMA = np.zeros([self.hiddenSize], np.float32)
        self.THETA = np.zeros([self.outputSize], np.float32)
        self.output_H = np.zeros([self.inputSize, self.hiddenSize], np.float32)
        self.output = np.zeros([self.outputSize], np.float32)
        self.Y = np.zeros([self.outputSize], float)
        self.X = np.zeros([self.inputSize], float)
        self.gradientHO = np.zeros([self.outputSize], np.float32)
        self.gradientIH = np.zeros([self.hiddenSize], np.float32)

        
    # calculate output
    def output(self, X):
        
        for h in range(self.hiddenSize):
            for i in range(self.inputSize):
                self.output_H[h] += self.V[i][h] * X[i]
            self.output_H[h] = sigmoid(self.output_H[h] - self.GAMMA[h])
        
        for j in range(self.outputSize):
            for h in range(self.hiddenSize):
                self.output[j] += self.W[h][j] * self.output_H[h]
            self.output[j] = sigmoid(self.output[j] - self.THETA[j])
        self.Y = self.output
        self.X = X

    # calculate gradient of weight between Hidden and Output
    def gradientHO(self, Y):
        
        for j in range(self.outputSize):
            self.gradientHO[j] = self.Y[j] * (1 - self.Y[j]) * (Y[j] - self.Y[j])
    
    # calculate gradient of weight between Input and Hidden
    def gradientIH(self):
        
        for h in range(self.hiddenSize):
            acc = 0.0
            for j in range(self.outputSize):
                acc += self.V[h][j] * self.gradientHO[j]
            self.gradientIH[h] = self.output_H[h] * (1 - self.output_H[h]) * acc

    def update(self):
        for j in range(self.outputSize):
            self.THETA[j] = self.THETA[j] - self.eta * self.gradientHO[j]
            for h in range(self.hiddenSize):
                self.W[h][j] = self.W[h][j] + self.eta * self.gradientHO[j] * self.output_H[h]
        for h in range(self.hiddenSize):
            self.GAMMA[h] = self.GAMMA[h] - self.eta * self.gradientIH[h]
            for i in range(self.inputSize):
                self.V[i][h] = self.V[i][h] + self.eta * self.gradientIH[h] * self.X

    def runFor1(self, X, Y):
        self.output(self, X)
        self.gradientHO(self, Y)
        self.gradientIH(self, Y)
        self.update(self)
    
    def test(self, X, Y):
        self.output(X)
        if(np.array_equal(self.Y, Y)):
            return True
        else:
            return False

