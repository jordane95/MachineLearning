from layers import Embedding, LSTM, TimeDense
from loss import TimeSoftmaxCrossEntropyLoss
from optimizer import SGD
from glove import get_glove_vec
from ptb_loader import read_data, make_batch, get_word_index
import numpy as np


MAX_EPOCHS = 5
BATCH_SIZE = 128
TIME_STEP = 60

EMBED_SIZE = 50
HIDDEN_SIZE = 32


class LSTMLm:
    def __init__(self, word_index, embed_size, hidden_size):
        vocab_size = len(word_index)
        We = get_glove_vec(save_path='./glove.6B.50d.txt', word_index=word_index, word_dim=embed_size)
        Wx = np.random.randn(embed_size, 4*hidden_size)
        Wh = np.random.randn(hidden_size, 4*hidden_size)
        b = np.zeros(4*hidden_size)
        Wl = .1*np.random.randn(hidden_size, vocab_size)
        bl = np.zeros(vocab_size)

        self.embed = Embedding(We)
        self.lstm = LSTM(Wx, Wh, b)
        self.linear = TimeDense(Wl, bl)
        self.loss = TimeSoftmaxCrossEntropyLoss()

        self.layers = [
            self.lstm,
            self.linear,
        ]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params.extend(layer.params)
            self.grads.extend(layer.grads)

    def forward(self, xs, ys):
        """

        :param xs: shape (N, T)
        :param ys: shape (N, T)
        :return:
        """
        xs = self.embed(xs)
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss(xs, ys)
        return loss

    def backward(self, dl=1):
        dxs = self.loss.backward(dl)
        dxs = self.linear.backward(dxs)
        dxs = self.lstm.backward(dxs)
        return dxs


def train():
    # loading training data
    data_batches, label_batches = make_batch(read_data('data/ptb.train'), batch_size=BATCH_SIZE, time_step=TIME_STEP)
    word_index = get_word_index()
    # build model
    lm = LSTMLm(word_index=word_index, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE)
    opt = SGD(learning_rate=0.1)
    num_batches = len(data_batches)
    for epoch in range(MAX_EPOCHS):
        loss = 0
        print("-"*10+"EPOCH"+str(epoch+1)+"-"*10)
        for i in range(num_batches):
            x = data_batches[i]
            y = label_batches[i]
            loss_ite = lm.forward(x, y)
            lm.backward()
            opt.update(lm.params, lm.grads)
            loss += loss_ite
            if i % 10 == 0: print("iteration %d, perplexity %d" % (i+1, np.exp(loss_ite)))
        loss /= num_batches
        print("One epoch finished")
        print("Epoch %d, perplexity %d" % (epoch + 1, np.exp(loss)))


if __name__ == '__main__':
    train()
