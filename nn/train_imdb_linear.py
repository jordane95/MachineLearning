from layers import Embedding, LSTM, Dense, TimeMean
from loss import SoftmaxCrossEntropyLoss, MSELoss
from optimizer import SGD
from trainer import Trainer
from glove import get_glove_vec
from data import load_imdb
from functions import softmax
import numpy as np


class SentiLSTM:
    def __init__(self, word_index, embed_size=50, output_size=2):
        """

        :param word_index: word to index dictionary
        :param embed_size: embedding size of word vectors, D = 50, 100, 200, 300 for glove
        :param output_size: binary
        """
        np.random.seed(10)
        # word vector look up matrix initialization
        We = get_glove_vec(save_path='glove/glove.6B.50d.txt', word_index=word_index, word_dim=embed_size)
        W = np.random.randn(embed_size, output_size)
        b = np.zeros(output_size)

        self.embed_layer = Embedding(We)
        self.pooling_layer = TimeMean()
        self.linear_layer = Dense(W, b)
        self.loss_layer = SoftmaxCrossEntropyLoss()

        self.layers = [
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

    def save_param(self, save_path='model/senti_linear.npz'):
        np.savez(save_path, W=self.params[0], b=self.params[1])
        print("Model parameters successfully saved !")
        return None

    def load_param(self, load_path='model/senti_linear.npz'):
        npz = np.load(load_path)
        self.params[0][...] = npz['W']
        self.params[1][...] = npz['b']
        print("Model parameters successfully loaded !")
        return None


def train_lstm_imdb():
    (train_data, train_labels), (test_data, test_labels), word_index = load_imdb()
    train_data, train_labels = train_data[:5000], train_labels[:5000]
    test_data, test_labels = test_data[:5000], test_labels[:5000]
    lstm = SentiLSTM(word_index=word_index, embed_size=50, output_size=2)
    opt = SGD(learning_rate=1, threshold=5)
    trainer = Trainer(model=lstm, optimizer=opt)
    trainer.fit(train_data, train_labels, test_data, test_labels, batch_size=32, epochs=100, decay=0.2)
    lstm.save_param(save_path='model/senti_linear_100.npz')
    opt.plot_norm(save_path='results/imdb/linear/')
    trainer.plot_losses(save_path='results/imdb/linear/sentiment_loss.jpg')
    trainer.plot_accuracy(save_path='results/imdb/linear/sentiment_accuracy.jpg')


def test():
    # to test if it can overfit
    (train_data, train_labels), (test_data, test_labels), word_index = load_imdb()
    train_data, train_labels = train_data[:10], train_labels[:10]
    test_data, test_labels = test_data[:100], test_labels[:100]
    lstm = SentiLSTM(word_index=word_index, embed_size=50, output_size=2)
    opt = SGD(learning_rate=0.001, threshold=5)
    trainer = Trainer(model=lstm, optimizer=opt)
    trainer.fit(train_data, train_labels, test_data, test_labels, batch_size=1, epochs=200, decay=0.2)


if __name__ == '__main__':
    train_lstm_imdb()
    # test()
