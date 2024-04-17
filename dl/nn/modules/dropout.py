from .module import Module
from ...tensor import Tensor
import numpy as np


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        assert 0 <= p < 1
        self.p = p

    def forward(self, x):
        if self.training:
            return x * np.random.binomial(1, 1 - self.p, x.shape[-1])
        return x * (1 - self.p)

    def __repr__(self):
        return "{}(p={})".format(self.__class__.__name__, self.p)