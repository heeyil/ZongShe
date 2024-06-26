import numpy as np
from .optimizer import Optimizer


class Nadam(Optimizer):  # 修改类名为 Nadam
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super().__init__(params)  # 使用父类 Optimizer 的初始化方法
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [np.zeros_like(param) for param in self.params]
        self.v = [np.zeros_like(param) for param in self.params]
        self.t = 1

    def step(self):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.weight_decay * self.params[i].data

            # 更新一阶动量m和二阶动量v
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad ** 2

            # 计算修正后的一阶动量m_hat和二阶动量v_hat
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # 更新参数
            self.params[i].data -= self.lr * (self.beta1 * m_hat + (1 - self.beta1) * grad) / (
                        np.sqrt(v_hat) + self.eps)

        self.t += 1