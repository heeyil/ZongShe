import numpy as np
from nn.Initializer import XavierUniform
from nn.Initializer import Zeros


class Layer(object):
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, num_in, num_out, w_init=XavierUniform(), b_init=Zeros()):
        super().__init__()

        self.params = {
            "w": w_init([num_in, num_out]),
            "b": b_init([1, num_out])}

        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad):
        self.grads["w"] = self.inputs.T @ grad
        self.grads["b"] = np.sum(grad, axis=0)
        return grad @ self.params["w"].T


class Activation(Layer):
    def __init__(self):
        super().__init__()
        self.inputs = None

    def _init_(self):
        super().__init__()
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.func(inputs)

    def backward(self, grad):
        return self.derivative_func(self.inputs) * grad

    def func(self, x):
        raise NotImplementedError

    def derivative_func(self, x):
        raise NotImplementedError


class ReLu(Activation):
    def __init__(self):
        super().__init__()

    def func(self, x):
        return np.maximum(x, 0.0)

    def derivative_func(self, x):
        return x > 0.0


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def func(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        return self.func(x) * (1.0 - self.func(x))


class tanh(Activation):
    def __init__(self):
        super().__init__()

    def func(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1.0 - self.func(x) ** 2


class LeakyReLu(Activation):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def func(self, x):
        x = x.copy()
        x[x < 0.0] *= self.gamma
        return x

    def derivative(self, x):
        dx = np.ones_like(x)
        dx[x < 0.0] = self.gamma
        return dx