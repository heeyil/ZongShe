from tensor import *
from typing import Callable, List, Optional, Tuple, Unio


"""
这个文件下所有的计算最后都应该是使用C++实现的
迫于技术不足
在此使用python实现
"""
def relu(input: Tensor) -> Tensor:
    return Tensor(tensor.maximum(0, input), device=input.device)


class sigmoid(tensor.UnaryOperator):
    def forward(self, input: Tensor) -> Tensor:
        sigmoid = self.xp.zeros(input.shape)
        sigmoid[input.data > 0] = 1 / (1 + self.xp.exp(-input.data[input.data > 0]))
        sigmoid[input.data <= 0] = 1 - 1 / (1 + self.xp.exp(input.data[input.data <= 0]))
        return Tensor(sigmoid, device=input.device)

    def grad_fn(self, input: Tensor, grad: np.ndarray) -> np.ndarray:
        return self.data * (1 - self.data) * grad


class tanh(tensor.UnaryOperator):
    def forward(self, x: Tensor) -> Tensor:
        tanh = self.xp.zeros(x.shape)
        tanh[x.data > 0] = 2 / (1 + self.xp.exp(-2 * x.data[x.data > 0])) - 1
        tanh[x.data <= 0] = 1 - 2 / (1 + self.xp.exp(2 * x.data[x.data <= 0]))
        return Tensor(tanh, device=input.device)

    def grad_fn(self, x: tensor.Tensor, grad: np.ndarray) -> np.ndarray:
        return (1 - self.data**2) * grad


class gelu:
    pass


def leaky_relu(input: Tensor, negative_slope: float) -> Tensor:
    return Tensor(tensor.maximum(input, negative_slope * input))


def softmax(x: tensor.Tensor, axis=None, keepdims=False)
