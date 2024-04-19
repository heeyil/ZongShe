from tensor import *
from typing import Callable, List, Optional, Tuple, Unio


"""
这个文件下所有的计算最后都应该是使用C++实现的
迫于技术不足
在此使用python实现
"""
def relu(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        return tensor.maximum
    else:
        return tensor.maximum(0, input)
        
