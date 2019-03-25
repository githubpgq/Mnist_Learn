import numpy as np
from progress_until import Progress


class knn:
    def __init__(self, k, images, label):
        self.k = k
        self.images = images
        self.label = label

    def knn(self, images, numToClassfy):
        result = np.zeros(numToClassfy, dtype=int)
        progress = Progress(numToClassfy, "KNN分类中")
        for i in range(numToClassfy):
            re = np.argmax(
                np.bincount(self.label[np.argpartition(
                    np.sum(np.square(self.images - images[i]), axis=1),
                    self.k)[0:self.k]]))
            result[i] = re
            #进度计算
            progress.updata(i + 1)
        return result

    def start(self, test_image, test_label, numToClassfy):
        count = 0
        result = self.knn(test_image, numToClassfy)
        for i in range(numToClassfy):
            if (result[i] == test_label[i]):
                count += 1
        print("KNN识别完成，样本个数: " + str(numToClassfy) + " 识别数: " + str(count) +
              " 正确率: %.2f %%" % ((count / numToClassfy) * 100))
        return result
