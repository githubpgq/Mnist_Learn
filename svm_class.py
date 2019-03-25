from libsvm.python.svmutil import *
from libsvm.python.svm import *


class Svm:
    def __init__(self, svm_label, svm_images, svm_test_label, svm_test_images):
        self.svm_label = svm_label
        self.svm_images = svm_images
        self.svm_test_label = svm_test_label
        self.svm_test_images = svm_test_images

    def train(self, numToClassfy, numToTrain, args):
        m = svm_train(self.svm_label[:numToTrain],
                      self.svm_images[:numToTrain], args)
        p_label, p_acc, p_val = svm_predict(
            self.svm_test_label[:numToClassfy],
            self.svm_test_images[:numToClassfy], m)
        return p_label