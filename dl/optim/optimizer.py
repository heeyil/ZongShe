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
    r"""Base class for all optimizers.
    
    Args:
        params:
        defaults: 字典类型传入的超参数
    """
    def __init__(self, params: ParamsT, defaults: Dict[str, Any]) -> None:
        self.defaults = defaults

        if isinstance(params, Tensor):
            raise TypeError(
                "params argument given to the optimizer should be "
                "an iterable of Tensors or dicts, but got " + torch.typename(params)
            )

        self.param_groups: List[Dict[str, Any]] = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        for param_group in param_groups:
            # cast将param_group强制转换为dict类型
            self.add_param_group(cast(dict, param_group))

    def zero_grad(self, set_to_none: bool = Fasle) -> None:
        r"""Resets the gradients of all optimized :class:`torch.Tensor` s.
        在反向传播计算梯度之前对上一次迭代时记录的梯度清零

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()
        
        @overload
    def step(self, closure: None = ...) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        r"""Performs a single optimization step (parameter update).

        Args:
            closure (Callable): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        .. note::
            Unless otherwise specified, this function should not modify the
            ``.grad`` field of the parameters.
        """
        raise NotImplementedError
            
    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        r"""Add a param group to the :class:`Optimizer` s `param_groups`.

        This can be useful when fine tuning a pre-trained network as frozen layers can be made
        trainable and added to the :class:`Optimizer` as training progresses.

        Args:
            param_group (dict): Specifies what Tensors should be optimized along with group
                specific optimization options.
        """
        if not isinstance(param_group, dict):
            raise TypeError(f"param_group must be a dict, but got {type(param_group)}")

        params = param_group["params"]
        if isinstance(params,Tensor):
            param_group["params"] = [params]
        elif isinstance(params, set):
            raise TypeError(
                "optimizer parameters need to be organized in ordered collections, but "
                "the ordering of tensors in sets will change between runs. Please use a list instead."
            )
        else:      
            param_group["params"] = list(params)

        for param in param_group["params"]:
            if not isinstance(param, Tensor):
                raise TypeError(
                    "optimizer can only optimize Tensors, "
                    "but one of the params is " + torch.typename(param)
                )
            if not self.defaults.get("differentiable", None) and not (
                param.is_leaf or param.retains_grad
            ):   
                raise ValueError("can't optimize a non-leaf Tensor")

        # 利用默认参数给所有组设置统一的超参数
        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    f"parameter group didn't specify a value of required optimization parameter {name}"
                )
            else:
                # setdefault(key, default=None)是字典方法
                # 如果字典中存在指定键'key'，则返回该键对应的值
                # 如果字典中不存在指定的键'key'，则将键'key'插入字典，并将其值设置为'default'参数指定的值，并返回该值。
                param_group.setdefault(name, default)

        params = param_group["params"]
        if len(params) != len(set(params)):
            warnings.warn(
                "optimizer contains a parameter group with duplicate parameters; "
                "in future, this will cause an error; "
                "see github.com/pytorch/pytorch/issues/40967 for more information",
                stacklevel=3,
            )

        # 将之前的所有参数整到这个param_set集合中
        param_set: Set[Tensor] = set()
        for group in self.param_groups:
            param_set.update(set(group["params"]))

        # isdisjoint()方法用于检查两个集合是否相交 -> set1.isdisjoint(set2)
        # 不相交 -> 返回True
        # 相交 -> 返回 False
        # 下面这个检查相交则raise ValueError
        if not param_set.isdisjoint(set(param_group["params"])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
