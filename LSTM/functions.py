import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def softmax(x):
    """
    batch level softmax
    :param x: shape (N, D), where N is the batch size, D is the dimension
    :return:
    """
    x_max = np.max(x, axis=1)
    x -= x_max.reshape((-1, 1))
    x = np.exp(x)
    x_sum = np.sum(x, axis=1)
    x /= x_sum.reshape((-1, 1))
    return x


def l2norm(x):
    return np.sqrt(np.sum(x**2))
