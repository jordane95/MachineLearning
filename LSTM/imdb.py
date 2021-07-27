from layers import Embedding, LSTM, Dense, TimeMean
from loss import SoftmaxCrossEntropyLoss, MSELoss
from optimizer import SGD
from trainer import Trainer
from glove import get_glove_vec
from data import load_imdb
from functions import softmax
import numpy as np


class SentiLSTM:
    def __init__(self, word_index, embed_size=50, hidden_size=64, output_size=2):
        """

        :param word_index: word to index dictionary
        :param embed_size: embedding size of word vectors, D = 50, 100, 200, 300 for glove
        :param hidden_size: LSTM hidden state dimension
        :param output_size: binary
        """
        np.random.seed(42)
        D, H, O = embed_size, hidden_size, output_size
        # word vector look up matrix initialization
        We = get_glove_vec(save_path='./glove.6B.50d.txt', word_index=word_index, word_dim=embed_size)
        # LSTM layer parameters initialization
        Wh = .1*np.random.randn(H, 4*H)
        Wx = .1*np.random.randn(D, 4*H)
        b = np.zeros(4*H)
        # Linear layer
        Wl = np.random.randn(H, O)
        bl = np.zeros(O)

        self.embed_layer = Embedding(We)
        self.lstm_layer = LSTM(Wx, Wh, b)
        self.pooling_layer = TimeMean()
        self.linear_layer = Dense(Wl, bl)
        self.loss_layer = SoftmaxCrossEntropyLoss()
        # self.loss_layer = MSELoss()

        self.layers = [
            self.lstm_layer,
            self.pooling_layer,
            self.linear_layer,
        ]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params.extend(layer.params)
            self.grads.extend(layer.grads)

    def forward(self, xs, ys):
        """

        :param xs: training samples, shape (N, T)
        :param ys: training labels, shape (N,)
        :return:
        """
        xs = self.embed_layer.forward(xs) # shape (N, T, D)
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ys)
        return loss

    def backward(self, dl=1):
        dxs = self.loss_layer.backward(dl)
        for layer in reversed(self.layers):
            dxs = layer.backward(dxs)
        return dxs

    def evaluate(self, xs, ys):
        """
        evaluate the accuracy of the model
        :param xs: shape (N, T)
        :param ys: shape (N,)
        :return:d
        """
        xs = self.embed_layer.forward(xs)
        for layer in self.layers:
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


def train_lstm_imdb():
    (train_data, train_labels), (test_data, test_labels), word_index = load_imdb()
    train_data, train_labels = train_data[:1000], train_labels[:1000]
    test_data, test_labels = test_data[:500], test_labels[:500],
    lstm = SentiLSTM(word_index=word_index, embed_size=50, hidden_size=32, output_size=2)
    opt = SGD(learning_rate=1, threshold=5)
    trainer = Trainer(model=lstm, optimizer=opt)
    trainer.fit(train_data, train_labels, test_data, test_labels, batch_size=32, epochs=100, decay=0.2)
    opt.plot_norm(save_path='images/imdb/')
    trainer.plot_losses(save_path='images/imdb/sentiment_loss_0.1.jpg')
    trainer.plot_accuracy(save_path='images/imdb/sentiment_accuracy_0.1.jpg')


if __name__ == '__main__':
    train_lstm_imdb()
