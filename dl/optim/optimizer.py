from math import sqrt
from tensor import Tensor
import functools
import math
import warnings
from collections import defaultdict, OrderedDict
from copy import deepcopy
from itertools import chain
from typing import (
    Any,
    Callable,
    cast,
    DefaultDict,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    overload,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from typing_extensions import ParamSpec, Self, TypeAlias


ParamsT: TypeAlias = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class Optimizer:
    """Base class for all optimizers."""
    def __init__(self, params: ParamsT, defaults: Dict[str, Any]) -> None:
        self.defaults = defaults

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
