from layers import Embedding, LSTM, TimeDense
from loss import TimeSoftmaxCrossEntropyLoss
from optimizer import SGD
from glove import get_glove_vec
from ptb_loader import read_data, make_batch, get_word_index
import numpy as np
import os


MAX_EPOCHS = 10
BATCH_SIZE = 128
TIME_STEP = 60

EMBED_SIZE = 50
HIDDEN_SIZE = 32


class LSTMLm:
    def __init__(self, word_index, embed_size, hidden_size):
        vocab_size = len(word_index)
        np.random.seed(42)
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

    def save_param(self, save_path='model/lm.npz'):
        np.savez(save_path,
                 Wx=self.params[0], Wh=self.params[1], b=self.params[2],
                 Wl=self.params[3], bl=self.params[4])
        print("Model parameters successfully saved !")
        return None

    def load_param(self, load_path='model/lm.npz'):
        npz = np.load(load_path)
        self.params[0][...] = npz['Wx']
        self.params[1][...] = npz['Wh']
        self.params[2][...] = npz['b']
        self.params[3][...] = npz['Wl']
        self.params[4][...] = npz['bl']
        print("Model parameters successfully loaded !")
        return None


def train():
    # loading training data
    data_batches, label_batches = make_batch(read_data('data/ptb.train'), batch_size=BATCH_SIZE, time_step=TIME_STEP)
    word_index = get_word_index()
    # load test data
    test_data_batches, test_label_batches = make_batch(read_data('data/ptb.test'), batch_size=BATCH_SIZE, time_step=TIME_STEP)
    # build model
    lm = LSTMLm(word_index=word_index, embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE)
    lm.load_param(load_path='model/lm.npz')
    perplexities = []
    test_pers = []
    opt = SGD(learning_rate=0.1)
    num_batches = len(data_batches)
    for epoch in range(MAX_EPOCHS):
        loss = 0
        # opt.lr = 1.0/(1.0+epoch)
        print("-"*10+"EPOCH"+str(epoch+1)+"-"*10)
        for i in range(num_batches):
            x = data_batches[i]
            y = label_batches[i]
            loss_ite = lm.forward(x, y)
            # perplexities.append(np.exp(loss_ite))
            # lm.backward()
            # opt.update(lm.params, lm.grads)
            loss += loss_ite
            if i % 10 == 0: print("iteration %d, perplexity %d" % (i+1, np.exp(loss_ite)))
        loss /= num_batches
        losses = [lm.forward(test_data, test_label) for test_data, test_label in zip(test_data_batches, test_label_batches)]
        test_loss = sum(losses)/len(losses)
        perplexities.append(np.exp(loss))
        test_pers.append(np.exp(test_loss))
        print("-"*10+"FINISH"+"-"*10)
        print("Epoch %d, train loss %f, test loss %f " % (epoch + 1, loss, test_loss))
    lm.save_param(save_path='model/lm_with_test_60.npz')
    import matplotlib.pyplot as plt
    plt.plot(perplexities, label='training perplexity')
    plt.plot(test_pers, label='test perplexity')
    plt.plot()
    plt.xlabel('epoch')
    plt.ylabel('perplexity')
    plt.legend()
    plt.savefig('results/ptb/perplexity_with_test_epoch50-60.jpg')


if __name__ == '__main__':
    train()
