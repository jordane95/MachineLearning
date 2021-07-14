import numpy as np


class SGD:
    def __init__(self, learning_rate=1e-3):
        self.lr = learning_rate

    def update(self, params, grads):
        assert len(params) == len(grads), "The shapes of parameters et gradients do not match !"
        for param, grad in zip(params, grads):
            param -= self.lr * grad
            import pprint
            # pprint.pprint(grad)
            # print(np.sum(grad > 1e-3))
            # after the update step, set the gradients to zero
            grad[...] = np.zeros_like(grad)
