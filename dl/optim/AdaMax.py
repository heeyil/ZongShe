import numpy as np
from .optimizer import Optimizer


class AdaMax(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(param) for param in self.params]
        self.v = [np.zeros_like(param) for param in self.params]
        self.t = 1

    def step(self):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.weight_decay * self.params[i].data

            # 更新一阶动量（m）和无穷范数（v）
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = np.maximum(self.beta2 * self.v[i], np.abs(grad))

            # 计算修正后的一阶动量（m_hat）
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)

            # 更新参数
            self.params[i].data -= self.lr * m_hat / (self.v[i] + self.eps)

        self.t += 1
