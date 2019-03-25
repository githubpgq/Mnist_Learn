import numpy as np


def activation_fun(input1):
    return  (1 / (1 + np.exp(-input1)))

def d_activation_fun(y):
    return y * (1 - y)
