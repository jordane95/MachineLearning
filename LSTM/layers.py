import numpy as np
from functions import sigmoid, softmax


class LSTMCell:
    def __init__(self, Wx, Wh, b):
        """
        LSTM Cell at single time step
        :param Wx: shape (D, 4H)
        :param Wh: shape (H, 4H)
        :param b: shape (4H,)
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        """
        forward pass
        :param x: shape (N, D)
        :param h_prev: shape (N, H)
        :param c_prev: shape (N, H)
        :return:
        """
        Wx, Wh, b = self.params
        H = Wx.shape[1]//4
        # print(h_prev)
        a = np.dot(x, Wx)+np.dot(h_prev, Wh)+b
        i, o, f, c_ = a[:, 0:H], a[:, H:2*H], a[:, 2*H:3*H], a[:, 3*H:]
        i, o, f = sigmoid(i), sigmoid(o), sigmoid(f)
        c_ = np.tanh(c_)
        # print(i.shape, c_.shape, f.shape, c_prev.shape)
        c = i*c_+f*c_prev
        c = np.tanh(c)
        h = o*c
        self.cache = (x, c, i, o, f, c_, c_prev, h_prev)
        return h, c

    def backward(self, dh, dc):
        """
        error back propagation
        :param dh: shape (N, H)
        :param dc: shape (N, H)
        :return:
        """
        Wx, Wh, b = self.params
        x, c, i, o, f, c_, c_prev, h_prev = self.cache
        do = dh*c
        dc += dh*o*(1-c**2)
        dc_prev = dc*f
        df = dc*c_prev
        di = dc*c_
        dc_ = dc*i
        dc_ *= (1-c_**2)
        di *= i*(1-i)
        do *= o*(1-o)
        df *= f*(1-f)
        da = np.hstack((di, do, df, dc_))
        dh_prev = np.dot(da, Wh.T)
        dWx = np.dot(x.T, da)
        dWh = np.dot(h_prev.T, da)
        db = np.sum(da, axis=0)
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        dx = np.dot(da, Wx.T)
        return dx, dh_prev, dc_prev


class LSTM:
    def __init__(self, Wx, Wh, b):
        """
        Recurrent LSTM
        :param Wx: shape (D, 4H)
        :param Wh: shape (H, 4H)
        :param b: shape (4H,)
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]

        self.layers = []

    def forward(self, xs):
        """
        forward pass of T steps
        :param xs: shape (N, T, D)
        :return:
        """
        N, T, D = xs.shape
        Wx, Wh, b = self.params
        H = b.shape[0]//4
        self.layers = []
        for t in range(T):
            self.layers.append(LSTMCell(Wx, Wh, b))
        hs = np.zeros((N, T, H))
        h, c = np.zeros((N, H)), np.zeros((N, H))
        for t, layer in enumerate(self.layers):
            xt = xs[:, t, :]
            h, c = layer.forward(xt, h, c)
            hs[:, t, :] = h
        return hs

    def backward(self, dhs):
        """
        backward propagation through time
        :param dhs: shape (N, T, H)
        :return:
        """
        Wx, Wh, b = self.params
        D = Wx.shape[0]
        N, T, H = dhs.shape
        dxs = np.zeros((N, T, D))

        dh = dhs[:, -1, :]
        dc = 0
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dh, dc)
            dxs[:, t, :] = dx
            dh += dhs[:, t-1, :]

        for layer in self.layers:
            for i in range(3):
                self.grads[i] += layer.grads[i]

        return dxs


class TimeMean:
    def __init__(self):
        """
        average of all the time steps
        """
        self.cache = None
        self.params = []
        self.grads = []

    def forward(self, xs):
        """

        :param xs: (N, T, H)
        :return:
        """
        N, T, H = xs.shape
        self.cache = (xs,)
        y = np.sum(xs, axis=1)/T
        return y

    def backward(self, dy):
        """

        :param dy: shape (N, H)
        :return:
        """
        xs, = self.cache
        N, T, H = xs.shape
        dx = dy/T
        dxs = np.zeros_like(xs)
        for t in range(T):
            dxs[:, t, :] = dx.copy()
        return dxs


class Dense:
    def __init__(self, W, b):
        """
        Linear layer
        :param W: shape (D, O)
        :param b: shape (O,)
        """
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.cache = None

    def forward(self, x):
        """
        forward pass
        :param x: shape (N, D)
        :return:
        """
        W, b = self.params
        self.cache = (x, )
        y = np.dot(x, W)+b
        return y

    def backward(self, dy):
        """
        backward pass
        :param dy: shape (N, O)
        :return:
        """
        W, b = self.params
        x, = self.cache
        dW = np.dot(x.T, dy)
        db = np.sum(dy, axis=0)
        self.grads[0] += dW
        self.grads[1] += db
        dx = np.dot(dy, W.T)
        return dx


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


def test_linear():
    W = np.random.randn(3, 4)
    b = np.random.randn(4)
    linear = Dense(W, b)
    xs = np.array([[1, 2, 3],
                   [2, 3, 4]])
    ys = np.array([[1, 2, 3, 4],
                   [2, 3, 4, 5]])
    from optimizer import SGD
    opt = SGD()
    for i in range(100):
        os = linear.forward(xs)
        loss = np.sum((ys-os)**2)/2
        print("epoch % d, loss %.2f" % (i+1, loss))
        dy = os-ys
        linear.backward(dy)
        opt.update(linear.params, linear.grads)


def test_mean():
    mean = TimeMean()
    print(mean.forward(np.ones((10, 5, 5))))
    print(mean.backward(dy=np.ones((5, 5))))


def test_softmax():
    softmax_layer = SoftmaxCrossEntropyLoss()


if __name__ == '__main__':
    test_linear()
    # test_mean()
