import numpy as np
from Activation import activation_fun


class Neuron():
    neuron_ouput = []
    neuron_deltes = []

    def __init__(self, inputNum, neuron_num):
        self.bias = -0.5 +  np.random.rand(neuron_num, 1)
        self.weight = -0.5 +  np.random.rand(neuron_num, inputNum)
        #self.weight[0] = -1
        #self.deltes = 0.0

    def getY(self, X):
        return activation_fun(np.dot(self.weight, X) + self.bias)
        #self.V = np.dot(np.insert(X, 0, -1), self.weight)
        #self.Y = activation_fun(1, self.V)
        #return self.Y

    def adjustWeight(self, y, yita, deltes):
        self.weight += (yita * y * deltes).T
        self.bias += (yita * deltes).T
        #print("afer adjust")
        #print(self.weight)
    def changeWeight(self, values, index):
        for i in range(len(values) - 1):
            self.weight[index][i] = values[i]
        self.bias[index][0] = values[len(values) - 1]
