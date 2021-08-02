from layers import LSTM, TimeMean, Dense
from loss import MSELoss, SoftmaxCrossEntropyLoss
from optimizer import SGD
from trainer import Trainer
from data import load_mnist
from functions import softmax
import numpy as np


class LSTMNet:
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        LSTM layer + linear layer at the final time step + softmax with cross entropy loss
        :param input_dim:
        :param hidden_dim:
        :param output_dim:
        """
        np.random.seed(42)
        D, H, O = input_dim, hidden_dim, output_dim
        Wx = np.random.randn(D, 4*H)
        Wh = np.random.randn(H, 4*H)
        b = np.zeros(4*H)
        Wl = np.random.randn(H, O)
        bl = np.zeros(O)
        self.lstm_layer = LSTM(Wx, Wh, b)
        self.mean_layer = TimeMean()
        self.linear_layer = Dense(Wl, bl)
        self.loss_layer = SoftmaxCrossEntropyLoss()
        # self.loss_layer = MSELoss()
        self.layers = [
            self.lstm_layer,
            self.mean_layer,
            self.linear_layer,
            self.loss_layer
        ]

        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params.extend(layer.params)
            self.grads.extend(layer.grads)

    def forward(self, xs, ys):
        """

        :param xs: shape (T, N, D)
        :param ys: shape (N, D), supervision labels
        :return:
        """
        for layer in self.layers[:-1]:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ys)
        return loss

    def backward(self, dy=1):
        """

        :param dy: (N, O)
        :return:
        """
        for layer in reversed(self.layers):
            dy = layer.backward(dy)
        return dy

    def evaluate(self, xs, ys):
        """
        evaluate the accuracy of the model
        :param xs: shape (N, T, D)
        :param ys: shape (N,)
        :return:d
        """
        for layer in self.layers[:-1]:
            xs = layer.forward(xs)
        xs = softmax(xs)
        correct = 0
        N, = ys.shape
        # print(xs.shape, ys.shape)
        for n in range(N):
            if xs[n].argmax() == ys[n]:
                correct += 1
        accuracy = correct/N
        return accuracy


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train, y_train = x_train[:10000], y_train[:10000]
    x_train, x_test = x_train / 255.0, x_test / 255.0 # critical for faster convergence
    # x_test, y_test = x_test[:100], y_test[:100]
    N, T, D = x_train.shape
    O = 10
    dummy = LSTMNet(input_dim=D, hidden_dim=50, output_dim=O)
    sgd = SGD(learning_rate=1)
    trainer = Trainer(model=dummy, optimizer=sgd)
    trainer.fit(x_train, y_train, x_test, y_test, epochs=100, batch_size=64, decay=0.2)
    trainer.plot_losses(save_path='results/mnist/loss_mnist_with_normalization.jpg')
    trainer.plot_accuracy(save_path='results/mnist/accuracy_mnist_with_normalization.jpg')
