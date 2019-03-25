import os
import struct
import numpy as np
from libsvm.python.commonutil import svm_read_problem


def load_data():
    images_path = "mnist/train-images.idx3-ubyte"
    labels_path = "mnist/train-labels.idx1-ubyte"
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def load_data_test():
    images_path = "mnist/t10k-images.idx3-ubyte"
    labels_path = "mnist/t10k-labels.idx1-ubyte"
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


def data_to_libsvm(images, labels):
    path = "mnist/images_libsvm_form"
    debugpath = "a.txt"
    try:
        file1 = open(path)  
        file1.close()
        print("libsvm格式的mnist训练集文件已生成，读取数据中。。。")
        return svm_read_problem(path)

    except IOError:
        print("libsvm格式的训练集文件未生成，开始生成数据。")
        with open(path, "w") as svmfile:
            for i in range(len(images)):
                svmfile.write(str(labels[i]) + " ")
                for j in range(len(images[i])):
                    if (images[i][j] != 0):
                        svmfile.write(
                            str(j + 1) + ":" + "%.7f" % (images[i][j] / 255) +
                            " ")
                svmfile.write("\n")
        return svm_read_problem(path)


def testdata_to_libsvm(images, labels):
    path = "mnist/test_images_libsvm_form"
    debugpath = "b.txt"
    try:
        file1 = open(path)
        file1.close()
        print("libsvm格式的mnist测试集文件已生成，读取数据中。。。")
        return svm_read_problem(path)

    except IOError:
        print("libsvm格式的测试集文件未生成，开始生成数据。")
        with open(path, "w") as svmfile:
            for i in range(len(images)):
                svmfile.write(str(labels[i]) + " ")
                for j in range(len(images[i])):
                    if (images[i][j] != 0):
                        svmfile.write(
                            str(j + 1) + ":" + "%.7f" % (images[i][j] / 255) +
                            " ")
                svmfile.write("\n")
        return svm_read_problem(path)
