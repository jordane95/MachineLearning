from functions import softmax
import numpy as np


class SoftmaxCrossEntropyLoss:
    def __init__(self):
        """
        softmax layer with cross entropy loss
        """
        self.cache = None
        self.params = []
        self.grads = []
        pass

    def forward(self, o, y):
        """
        :param o: model output, shape (N, O)
        :param y: true label, shape (N,), where 0<=y[i]<O
        :return:
        """
        # print(o.shape, y.shape)
        y_ = softmax(o)
        self.cache = (y_, y)
        N, = y.shape
        # print(N)
        loss = -sum(np.log(y_[np.arange(N), y]))/N
        return loss

    def backward(self, dl=1):
        y_, y = self.cache
        N, D = y_.shape
        dl /= N
        do = y_.copy()
        for n in range(N):
            do[n][y[n]] -= 1
        do *= dl
        return do


class TimeSoftmaxCrossEntropyLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.cache = None
        self.layers = []

    def __call__(self, xs, ys):
        return self.forward(xs, ys)

    def forward(self, xs, ys):
        """

        :param xs: (N, T, V), predicted probability distribution
        :param ys: (N, T), true label
        :return:
        """
        N, T, V = xs.shape
        self.cache = xs.shape
        self.layers = []
        loss = 0
        for t in range(T):
            layer = SoftmaxCrossEntropyLoss()
            loss += layer.forward(xs[:, t, :], ys[:, t])
            self.layers.append(layer)
        loss /= T
        return loss

    def backward(self, dl=1):
        N, T, V = self.cache
        dl /= T
        dxs = np.zeros(self.cache)
        for t in range(T):
            dxs[:, t, :] = self.layers[t].backward(dl)
        return dxs


class MSELoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.cache = None

    def forward(self, o, y):
        """

        :param o: model output, shape (N, O)
        :param y: supervision label, shape (N,)
        :return:
        """
        N, = y.shape
        y_ = np.zeros_like(o)
        for n in range(N):
            y_[n][y[n]] = 1
        loss = np.sum((o-y_)**2)/2
        loss /= N
        self.cache = (o, y_)
        return loss

    def backward(self, dy=1):
        o, y_ = self.cache
        N, O = o.shape
        do = o-y_
        do /= N
        return do
