import numpy as np
from progress_until import Progress


class Bayes:
    def __init__(self, class_num, input_num, images, label):
        self.class_num = class_num
        self.input_num = input_num
        self.images = images
        self.label = label

    def train(self):
        prior = np.zeros(self.class_num)
        condition = np.zeros((self.class_num, self.input_num, 2))
        progress = Progress(len(self.images), "朴素贝叶斯训练中")

        for i in range(len(self.images)):
            prior[self.label[i]] += 1
            for j in range(self.input_num):
                condition[self.label[i]][j][self.images[i][j]] += 1
            progress.updata(i + 1)

        for i in range(self.class_num):
            for j in range(self.input_num):
                num_0 = condition[i][j][0]
                num_1 = condition[i][j][1]
                p0 = (float(num_0) / float(num_0 + num_1))
                p1 = (float(num_1) / float(num_0 + num_1))
                condition[i][j][0] = p0
                condition[i][j][1] = p1

        self.prior = prior
        self.condition = condition

    def start(self, images, label, numToClassfy):
        result = np.zeros(numToClassfy, dtype=int)
        progress = Progress(numToClassfy, "朴素贝叶斯分类中")
        count = 0
        self.train()
        for i in range(numToClassfy):
            max = 0
            index = 0
            for j in range(10):
                p = self.prior[j]
                for k in range(self.input_num):
                    p *= self.condition[j][k][images[i][k]]
                if (p > max):
                    max = p
                    index = j
            result[i] = index
            if (index == label[i]):
                count += 1
            progress.updata(i + 1)
        print("朴素贝叶斯识别完成，样本个数: " + str(numToClassfy) + " 识别数: " + str(count) +
              " 正确率: %.2f %%" % ((count / numToClassfy) * 100))
        return result
