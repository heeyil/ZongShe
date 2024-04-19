from .module import Module
from .. import functional as F
from typing import Optional, Tuple
from .. import constant_, xavier_normal_, xavier_uniform_
from parameter import Parameter
# import niu_zhe_yi as ___

__all__ = ['ReLU', 'Sigmoid', 'Tanh', 'GELU', 'LeakyReLu', 'Softmax', 'LogSoftmax']

"""
实际上全部的：
__all__ = ['Threshold', 'ReLU', 'RReLU', 'Hardtanh', 'ReLU6', 'Sigmoid', 'Hardsigmoid', 'Tanh',
           'SiLU', 'Mish', 'Hardswish', 'ELU', 'CELU', 'SELU', 'GLU', 'GELU', 'Hardshrink', 'LeakyReLU',
           'LogSigmoid', 'Softplus', 'Softshrink', 'MultiheadAttention', 'PReLU', 'Softsign', 'Tanhshrink',
           'Softmin', 'Softmax', 'Softmax2d', 'LogSoftmax']
"""


class ReLU(Module):
    r"""
    
    Args:
        inplace: 控制是否原地执行一个操作
                -> True:直接在原始内存位置上进行操作
                -> False:在新的内存位置上进行操作
                Default: ``False``

    Examples::

        >>> m = nn.ReLU()
        >>> input = randn(2)
        >>> output = m(input)
        >>> print(input)
        >>> print(output)

    在上面的例子中:
    假设input的初始值为[-1, 2]
    inplace=True -> 打印的input为[0, 2], output为[0, 2]
    inplace=False -> 打印的input为[-1, 2], output为[0, 2]

    总的来说，就是为True的话，原地执行操作，被操作变量与操作得到的变量的内存与值是绑定在一起的
    output的改变会引起input的改变
    False的话，output与input就是分开的，二者互不影响
    
    """

    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class Sigmoid(Module):
    r"""
    
    \frac{1}{1 + \exp(-x)}

    Examples::

        >>> m = nn.Sigmoid()
        >>> input = randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.sigmoid(input)


class Tanh(Module):
    r"""
    
    \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    Examples::

        >>> m = nn.Tanh()
        >>> input = randn(2)
        >>> output = m(input)
    """

    def forward(self, input: Tensor) -> Tensor:
        return F.tanh(input)


class GELU(Module):
    r"""
    
    GELU(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))

    Args:
        approximate (str, optional): the gelu approximation algorithm to use:
            ``'none'`` | ``'tanh'``. Default: ``'none'``
            
    Examples::

        >>> m = nn.GELU()
        >>> input = randn(2)
        >>> output = m(input)
    """

    __constants__ = ['approximate']
    approximate: str

    def __init__(self, approximate: str = 'none') -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, input: Tensor) -> Tensor:
        return F.gelu(input, approximate=self.approximate)

    def extra_repr(self) -> str:
        return f'approximate={repr(self.approximate)}'


class LeakyReLU(Module):
    r"""
    
    \max(0, x) + \text{negative\_slope} * \min(0, x)

    Args:
        negative_slope: Controls the angle of the negative slope (which is used for
          negative input values). Default: 1e-2
        inplace
        
    Examples::

        >>> m = nn.LeakyReLU(0.1)
        >>> input = randn(2)
        >>> output = m(input)
    """

    __constants__ = ['inplace', 'negative_slope']
    inplace: bool
    negative_slope: float

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.leaky_relu(input, self.negative_slope, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return f'negative_slope={self.negative_slope}{inplace_str}'


class Softmax(Module):
    r"""Applies the Softmax function to an n-dimensional input Tensor.

    Rescales them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1.

    Softmax is defined as:
    
    Softmax(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}


    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [0, 1]

    Args:
        dim (int): A dimension along which Softmax will be computed (so every slice
            along dim will sum to 1).

    Examples::

        >>> m = nn.Softmax(dim=1)
        >>> input = randn(2, 3)
        >>> output = m(input)

    """

    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return F.softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class LogSoftmax(Module):
    r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional input Tensor.

    The LogSoftmax formulation can be simplified as:

    .. math::
        \text{LogSoftmax}(x_{i}) = \log\left(\frac{\exp(x_i) }{ \sum_j \exp(x_j)} \right)

    Shape:
        - Input: :math:`(*)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(*)`, same shape as the input

    Args:
        dim (int): A dimension along which LogSoftmax will be computed.

    Returns:
        a Tensor of the same dimension and shape as the input with
        values in the range [-inf, 0)

    Examples::

        >>> m = nn.LogSoftmax(dim=1)
        >>> input = torch.randn(2, 3)
        >>> output = m(input)
    """

    __constants__ = ['dim']
    dim: Optional[int]

    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input: Tensor) -> Tensor:
        return F.log_softmax(input, self.dim, _stacklevel=5)

    def extra_repr(self):
        return f'dim={self.dim}'
