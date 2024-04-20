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


def softmax(input: Tensor, dim=None, keepdims=False) -> Tensor:
    input_sub_max = input - input.data.max()
    exp_ = tensor.exp(input_sub_max)
    softmax =  exp_ / tensor.sum(exp_, axis=dim, keepdims=keepdims)
    return Tensor(softmax, device=input.device)


def log_softmax(input: Tensor, dim=None, keepdims=keepdims) -> Tensor:
    input_sub_max = input - input.data.max()
    log_softmax = input_sub_max - tensor.log(
        tensor.sum(tensor.exp(input_sub_max), axis=dim, keepdims=keepdims))
    return Tensor(log_softmax)


def l1_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    loss = tensor.abs(input - target)
    if reduction == 'mean':
        return Tensor(tensor.mean(loss), device=input.device)
    elif reduction == 'sum':
        return Tensor(tensor.sum(loss), device=input.device)
    else:
        assert 0, "reduction must be mean or sum."


def nll_loss(
    input: Tensor, 
    target: Tensor, 
    reduction: str = 'mean',
) -> Tensor:
    nll = -input * target
    if reduction == 'mean':
        return Tensor(tensor.mean(nll), device=input.device)
    elif reduction == 'sum':
        return Tensor(tensor.sum(nll), device=input.device)
    else:
        assert 0, "reduction must be mean or sum."


def mse_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    square_sum = tensor.square(input - target)
    if reduction == 'mean':
        return Tensor(tensor.mean(square_sum), device=input.device)
    elif reduction == 'sum':
        return Tensor(tensor.sum(square_sum),device=input.device)
    else:
        assert 0, "reduction must be mean or sum."


def cross_entropy(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    update_input = input - input.data.max()
    log_sum_exp = tensor.log(
        tensor.sum(tensor.exp(update_input), 1, keepdims=True))
    nll = (log_sum_exp - update_input) * target
    if reduction == 'mean':
        return Tensor(tensor.mean(nll), device=input.device)
    elif reduction == 'sum':
        return Tensor(tensor.sum(nll), device=input.device)
    else:
        assert 0, "reduction must be mean or sum."
