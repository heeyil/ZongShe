import numpy as np


class Optimizer(object):
    def __init__(self, lr, epsilon, weight_decay):
        self.lr = lr
        self.weight_decay = weight_decay
        self.epsilon = epsilon

    def update(self, grads, params):
        step = list()
        flatten_grads = np.concatenate(
            [np.ravel(v) for grad in grads for v in grad.values()])

        flatten_step = self.compute_step(flatten_grads)

        p = 0
        for param in params:
            layer = dict()
            for k, v in param.items():
                block = np.prod(v.shape)
                _step = flatten_step[p:p + block].reshape(v.shape)
                _step -= self.weight_decay * v
                layer[k] = _step
                p += block
            step.append(layer)
        return step

    def compute_step(self, grad):
        raise NotImplementedError


class Momentum(Optimizer):
    def __init__(self, lr=1e-3, momentum=0.9, weight_decay=0.0, epsilon=0.0):
        super().__init__(lr, epsilon, weight_decay)
        self._momentum = momentum
        self._v = 0.0

    def compute_step(self, grad):
        self._v = self._momentum * self._v + grad
        step = -self._v * self.lr

        return step


class AdaGrad(Optimizer):
    def __init__(self, lr=1e-3, epsilon=1e-6, weight_decay=0.0):
        super().__init__(lr, epsilon, weight_decay)
        self.s = 0.0

    def compute_step(self, grad):
        self.s = self.s + grad ** 2
        adjust_lr = self.lr / (self.s + self.epsilon) ** 0.5
        step = - adjust_lr * grad

        return step


class Adadelta(Optimizer):
    def __init__(self, lr=1e-3, rho=0.9, epsilon=1e-6, weight_decay=0.0):
        super().__init__(lr, epsilon, weight_decay)
        self.rho = rho
        self.s = 0
        self.delta = 0

    def compute_step(self, grad):
        self.s = self.rho * self.s + (1 - self.rho) * (grad ** 2)
        std = (self.delta + self.epsilon) ** 0.5
        adjust_grad = grad * (std / (self.s + self.epsilon) ** 0.5)
        self.delta = self.rho * self.delta + (1 - self.rho) * (adjust_grad ** 2)
        step = -adjust_grad

        return step


class RMSProp(Optimizer):
    def __init__(self, lr=1e-3, gamma=0.9, epsilon=1e-6, weight_decay=0.0):
        super().__init__(lr, epsilon, weight_decay)
        self.gamma = gamma
        self.s = 0.0

    def compute_step(self, grad):
        self.s = self.gamma * self.s + (1 - self.gamma) * (grad ** 2)
        adjust_lr = self.lr / ((self.s + self.epsilon) ** 0.5)
        step = -adjust_lr * grad

        return step


class Adam(Optimizer):
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-6, weight_decay=0.0):
        super().__init__(lr, epsilon, weight_decay)
        self.beta1, self.beta2 = beta1, beta2

        self.t = 0
        self.v, self.s = 0, 0

    def compute_step(self, grad):
        self.t += 1
        self.v = self.beta1 * self.v + (1 - self.beta1) * grad
        self.s = self.beta2 * self.s + (1 - self.beta2) * (grad ** 2)

        v = self.v / (1 - self.beta1 ** self.t)
        s = self.s / (1 - self.beta2 ** self.t)
        step = -self.lr * v / (s ** 0.5 + self.epsilon)

        return step