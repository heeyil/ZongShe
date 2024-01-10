from .module import Module
from .. import functional as F


class Sigmoid(Module):
    def forward(self, x):
        return F.sigmoid(x)


class Tanh(Module):
    def forward(self, x):
        return F.tanh(x)


class ReLu(Module):
    def forward(self, x):
        return F.relu(x)


class LeakyReLu(Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x):
        return F.leaky_relu(x, self.alpha)


class Softmax(Module):
    def __init__(self, axis=None):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return F.softmax(x, self.axis)
