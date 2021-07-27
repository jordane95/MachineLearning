import numpy as np
import matplotlib.pyplot as plt
from functions import l2norm


class SGD:
    def __init__(self, learning_rate=.1, threshold=float('inf')):
        self.lr = learning_rate
        self.threshold = threshold
        self.grad_norm = [[], [], [], [], []]

    def update(self, params, grads):
        assert len(params) == len(grads), "The shapes of parameters et gradients do not match !"
        for i in range(len(params)):
            grad = grads[i]
            norm = l2norm(grad)
            self.grad_norm[i].append(norm)
            if norm > self.threshold:
                grad = grad * self.threshold / norm
            params[i] -= self.lr * grad
            import pprint
            # pprint.pprint(grad)
            # print(np.sum(grad > 1e-3))
            # after the update step, set the gradients to zero
            grads[i][...] = np.zeros_like(grad)

    def plot_norm(self, save_path='results/'):
        for i in range(5):
            plt.plot(self.grad_norm[i], label='gradient l2 norm')
            plt.xlabel('steps')
            plt.ylabel('norm')
            plt.legend()
            plt.savefig(save_path+'grad_norm_l2_'+str(i)+'.jpg')
            plt.show()
