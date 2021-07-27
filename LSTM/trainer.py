import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        self.losses = []
        self.train_accs = []
        self.test_accs = []

    def fit(self, x_train, y_train, x_test, y_test, epochs=5, batch_size=64, decay=None):
        """
        training process
        :param x_train: input of training set, shape (N, T, D)
        :param y_train: label of training set, shape (N,)
        :param x_test: test set for evaluation
        :param y_test: test set for evaluation
        :param epochs: training epoch number
        :param batch_size: batch size of an iteration
        :param decay: learning rate decay
        :return:
        """
        total_size, = y_train.shape
        self.losses = []
        self.train_accs = []
        self.test_accs = []
        for epoch in range(epochs):
            permutation = np.random.permutation(total_size)
            shuffled_x_train = x_train[permutation]
            shuffled_y_train = y_train[permutation]
            iterations = total_size//batch_size
            loss = 0
            for iteration in range(iterations):
                x_batch = shuffled_x_train[batch_size*iteration:batch_size*(iteration+1)]
                # print(x_batch.shape)
                y_batch = shuffled_y_train[batch_size*iteration:batch_size*(iteration+1)]
                # print(x_batch.shape, y_batch.shape)
                loss_iter = self.model.forward(x_batch, y_batch)
                loss += loss_iter
                self.model.backward()
                self.optimizer.update(self.model.params, self.model.grads)
                # print("Epoch %d, iter %d, loss %.2f" % (epoch+1, iteration+1, loss_iter))
            # Performance evaluation per epoch
            loss /= iterations
            train_acc = self.model.evaluate(x_train, y_train) # evaluate performance on training set
            test_acc = self.model.evaluate(x_test, y_test) # evaluate performance on test set
            if decay: self.optimizer.lr = 1/(1+decay*epoch) # learning rate decay
            print("Epoch %d, loss %.3f, train accuracy %.3f, test accuracy %.3f" % (epoch+1, loss, train_acc, test_acc))
            self.losses.append(loss)
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)

    def plot_losses(self, save_path="./results/loss.jpg"):
        plt.plot(self.losses, label='training loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(save_path)
        plt.show()

    def plot_accuracy(self, save_path="./results/accuracy.jpg"):
        plt.plot(self.train_accs, label='training accuracy')
        plt.plot(self.test_accs, label='test accuracy')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(save_path)
        plt.show()


