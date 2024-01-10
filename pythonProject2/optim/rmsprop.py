from .optimier import Optimizer
import numpy as np


class Adadelta(Optimizer):
    def __init__(
            self,
            params,
            lr,
            gamma=float(0.9),
            weight_decay=float(0),
            eps=float(1e-6),
    ) -> None:
        super().__init__(params)
        self.lr = lr
        self.gamma = gamma
        self.eps = eps
        self.weight_decay = weight_decay
        self.s = [np.zeros(param.shape) for param in self.params]

    def step(self):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.weight_decay * self.params[i].data

            self.s[i] = self.gamma * self.s[i] + (1 - self.gamma) * grad ** 2
            self.params[i].data -= self.lr * grad / (self.s[i] + self.eps) ** 0.5