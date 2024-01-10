from .optimier import Optimizer
import numpy as np


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas= (0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.v = [np.zeros(param.shape) for param in self.params]
        self.s = [np.zeros(param.shape) for param in self.params]
        self.t = 1

    def step(self):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.weight_decay * self.params[i].data
            self.v[i] = self.beta1 * self.v[i] + (1 - self.beta1) * grad
            self.s[i] = self.beta2 * self.s[i] + (1 - self.beta2) * grad**2

            v = self.v[i] / (1 - self.beta1**self.t)
            s = self.s[i] / (1 - self.beta2**self.t)

            self.params[i].data -= self.lr * v / (
                s**0.5 + self.eps)
        self.t += 1