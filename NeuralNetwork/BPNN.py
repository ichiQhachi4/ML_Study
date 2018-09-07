
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
        
        self.hiddenSize = 3
        self.eta = 0.1

        # initial weight with random variables
        self.V = np.random.randn(self.inputSize, self.hiddenSize)
        self.W = np.random.randn(self.hiddenSize, self.outputSize)
        self.GAMMA = np.zeros((1, self.hiddenSize))    # 列向量
        self.THETA = np.zeros((1, self.outputSize))
        self.output_H = np.zeros((1, self.hiddenSize)) # 列向量
        self.output = np.zeros((1, self.outputSize))
        self.Y = np.zeros((1, self.outputSize))
        self.X = np.zeros((1, self.inputSize))
        self.gradientHO = np.zeros((1, self.outputSize))
        self.gradientIH = np.zeros((1, self.hiddenSize))


        
    # calculate output
    def output_ex(self):
        output_H_temp = np.dot(self.X, self.V)
        self.output_H = sigmoid_v(output_H_temp - self.GAMMA)
        '''
        for h in range(self.hiddenSize):
            for i in range(self.inputSize):
                self.output_H[h] += self.V[i][h] * X[i]
            self.output_H[h] = sigmoid_v(self.output_H[h] - self.GAMMA[h])
        '''
        output_temp = np.dot(self.output_H, self.W)
        self.Y = sigmoid_v(output_temp - self.THETA)

        '''
        for j in range(self.outputSize):
            for h in range(self.hiddenSize):
                self.output[j] += self.W[h][j] * self.output_H[h]
            self.output[j] = sigmoid_v(self.output[j] - self.THETA[j])
        '''
    
    def generate(self, X):
        output_H = np.dot(self.X, self.V)
        output_H = sigmoid_v(output_H - self.GAMMA)
        output = np.dot(output_H, self.W)
        return sigmoid_v(output - self.THETA)



    # calculate gradient of weight between Hidden and Output
    def gradientHO_ex(self, Y):
        # print(self.Y)
        # print(Y)
        
        #self.gradientHO = gradient_v(self.Y, Y)
        # the vectorize function can't be applied to two vector!!!
        for j in range(self.outputSize):
            self.gradientHO[0][j] = self.Y[0][j] * (1 - self.Y[0][j]) * (Y[j] - self.Y[0][j])
        
    


    # calculate gradient of weight between Input and Hidden
    def gradientIH_ex(self):
        # acc = self.W * self.gradientHO
        # self.gradientIH = gradient2_v(self.output_H, acc)
        
        for h in range(self.hiddenSize):
            acc = 0.0
            for j in range(self.outputSize):
                acc += self.W[h][j] * self.gradientHO[0][j]
            self.gradientIH[0][h] = self.output_H[0][h] * (1 - self.output_H[0][h]) * acc


    def update(self):
        self.THETA = self.THETA - self.eta * self.gradientHO
        self.GAMMA = self.GAMMA - self.eta * self.gradientIH
        self.W = self.W + self.eta * np.dot(self.gradientHO.reshape(self.hiddenSize,1), self.output_H.reshape(1,self.outputSize))
        self.V = self.V + self.eta * np.dot(self.X.reshape(self.inputSize,1), self.gradientIH.reshape(1,self.hiddenSize))
        '''
        for j in range(self.outputSize):
            self.THETA[j] = self.THETA[j] - self.eta * self.gradientHO[j]
            for h in range(self.hiddenSize):
                self.W[h][j] = self.W[h][j] + self.eta * self.gradientHO[j] * self.output_H[h]
        for h in range(self.hiddenSize):
            self.GAMMA[h] = self.GAMMA[h] - self.eta * self.gradientIH[h]
            for i in range(self.inputSize):
                self.V[i][h] = self.V[i][h] + self.eta * self.gradientIH[h] * self.X[i]
        '''

    def run(self):
        for i in range(self.times):
            self.X = self.X_train[i]
            self.output_ex()
            self.gradientHO_ex(self.Y_train[i])
            self.gradientIH_ex()
            self.update()

            
    def test(self, X, Y):
        Y_o = np.transpose(self.generate(X))

        result = True
        for i in range(self.outputSize):
            if abs(Y[i] - Y_o[i]) >= 0.1:
                print (Y)
                print(Y_o)
                return False
        
        return result


