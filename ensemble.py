import numpy as np
from knn import knn
from neural_network import Network
from bayes import Bayes
from svm_class import Svm
from data_input import load_data, load_data_test, data_to_libsvm, testdata_to_libsvm
from progress_until import Progress

numToTrain = 60000
numToClassfy = 10000
model_num = 5
model_now = 0

results = np.zeros([model_num, numToClassfy], dtype=int)

print("开始读取MNIST数据：")
images, label = load_data()
test_images, test_label = load_data_test()
#libsvm格式数据读取
svm_label, svm_images = data_to_libsvm(images, label)
svm_test_label, svm_test_images = testdata_to_libsvm(test_images, test_label)

#数据二值化
images = np.where(images >= 128, 1, 0)
test_images = np.where(test_images >= 128, 1, 0)

network = Network(784, 2, [200, 10], 10)
network.startLearing(numToTrain, images, label, 1)
results[model_now] = network.test(test_images, test_label, numToClassfy)
model_now += 1

print("SVM 1训练中...")
svm = Svm(svm_label, svm_images, svm_test_label, svm_test_images)
results[model_now] = svm.train(numToClassfy, numToTrain, "-q -m 1000")
model_now += 1

print("SVM 2训练中...")
svm = Svm(svm_label, svm_images, svm_test_label, svm_test_images)
results[model_now] = svm.train(numToClassfy, numToTrain, "-q -m 1000 -t 3")
model_now += 1

knn = knn(10, images, label)
results[model_now] = knn.start(test_images, test_label, numToClassfy)
model_now += 1

bayes = Bayes(10, 784, images, label)
results[model_now] = bayes.start(test_images, test_label, numToClassfy)
model_now += 1

results = results.T
count = 0
for i in range(numToClassfy):
    progress = Progress(numToClassfy, "正在投票")
    k = np.argmax(np.bincount(results[i]))
    if (k == test_label[i]):
        count += 1
    progress.updata(i + 1)
print("集成识别完成，样本个数: " + str(numToClassfy) + " 识别数: " + str(count) +
      " 正确率: %.2f %%" % ((count / numToClassfy) * 100))
