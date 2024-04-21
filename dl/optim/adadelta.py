from .optimier import Optimizer
import numpy as np
from typing import List, Optional
from tensor import Tensor


class Adadelta(Optimizer):
    def __init__(
        self,
        params,
        lr=1.0,
        rho=0.9,
        eps=1e-6
        weight_decay=0,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
        )

        super().__init__(params, defaults)

    def step(self, closure=None):
        for i in range(len(self.params)):
            grad = self.params[i].grad + self.weight_decay * self.params[i].data

            self.s[i] = self.rho * self.s[i] + (1 - self.rho) * grad ** 2
            adjust_grad = ((self.delta[i] + self.eps) ** 0.5) * grad / (self.s[i] + self.eps) ** 0.5
            self.delta[i] = self.rho * self.delta[i] + (1 - self.rho) * adjust_grad ** 2
            self.params[i].data -= adjust_grad
