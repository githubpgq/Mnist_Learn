from neuron import Neuron
import numpy as np
from Activation import d_activation_fun
from progress_until import Progress


class Network():
    def __init__(self, inputNum, hiddenLayerNum, neuronNum, outputNum):
        self.inputNum = inputNum
        self.hiddenLayerNum = hiddenLayerNum
        self.outputNum = outputNum
        self.neuronNum = neuronNum

        self.yita = 0.4

        self.layers = []
        self.initilize()

    def initilize(self):
        weight_num = self.inputNum
        for i in range(self.hiddenLayerNum):
            self.layers.append(Neuron(weight_num, self.neuronNum[i]))
            weight_num = self.neuronNum[i]
            #self.deltes.append(list(temp2))
        for i in range(self.hiddenLayerNum):
            Y = np.zeros(self.neuronNum[i])
            Y2 = np.zeros(self.neuronNum[i])
            Neuron.neuron_ouput.append(Y)
            Neuron.neuron_deltes.append(Y2)

    def train(self, labels, images):
        #输入层到隐藏层
        Neuron.neuron_ouput[0] = self.layers[0].getY(images)
        #隐藏层到输出层
        for i in range(1, len(self.layers)):
            Neuron.neuron_ouput[i] = self.layers[i].getY(
                Neuron.neuron_ouput[i - 1])
        #反向
        d = np.zeros([self.outputNum, 1])
        d[labels] = 1

        output_layer = self.hiddenLayerNum - 1
        Neuron.neuron_deltes[output_layer] = (
            (d - Neuron.neuron_ouput[output_layer]) * d_activation_fun(
                Neuron.neuron_ouput[output_layer])).T

        for i in range(len(self.layers) - 1)[::-1]:
            Neuron.neuron_deltes[i] = np.dot(
                Neuron.neuron_deltes[i + 1],
                self.layers[i + 1].weight) * d_activation_fun(
                    Neuron.neuron_ouput[i]).T
            self.layers[i + 1].adjustWeight(Neuron.neuron_ouput[i], self.yita,
                                            Neuron.neuron_deltes[i + 1])
        self.layers[0].adjustWeight(images, self.yita, Neuron.neuron_deltes[0])

    # def adjust(self, images):
    #     for i in range(len(self.layers[0])):
    #         self.layers[0][i].adjustWeight(images, self.yita,
    #                                        Neuron.neuron_deltes[0][i])
    #     for i in range(1, len(self.layers)):
    #         for j in range(len(self.layers[i])):
    #             self.layers[i][j].adjustWeight(Neuron.neuron_ouput[i - 1],
    #                                            self.yita,
    #                                            Neuron.neuron_deltes[i][j])

    def startLearing(self, sampleNums, images, labels, times):
        progress = Progress(sampleNums * times, "BP网络训练中")
        for n in range(times):
            for i in range(sampleNums):
                self.train(labels[i], images[i].reshape(self.inputNum, 1))
                progress.updata(i + sampleNums * n + 1)

    def get_output(self):
        return Neuron.neuron_ouput[self.hiddenLayerNum - 1]

    # def test(self, imags, labels, nums):
    #     fit_num = 0
    #     for i in range(nums):
    #         self.forward(imags[i])
    #         y = self.get_output()[0]
    #         print(y)
    def forward_only(self, images):
        Neuron.neuron_ouput[0] = self.layers[0].getY(images)
        for i in range(1, len(self.layers)):
            Neuron.neuron_ouput[i] = self.layers[i].getY(
                Neuron.neuron_ouput[i - 1])

    def test(self, images, labels, numToClassfy):
        result = np.zeros(numToClassfy,dtype=int)
        count = 0
        for i in range(numToClassfy):
            self.forward_only(images[i].reshape(self.inputNum, 1))
            y = self.get_output()
            m = np.argmax(y.T)
            result[i] = m
            #print(y)
            if (labels[i] == m):
                count += 1
        print("BP网络识别完成，样本个数: " + str(numToClassfy) + " 识别数: " + str(count) +
              " 正确率: %.2f %%" % ((count / numToClassfy) * 100))
        return result
