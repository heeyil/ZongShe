from collections import OrderedDict
from nn.parameter import Parameter
from cuda import Device, current_device
from autograd import set_grad_enabled

class Module:
    r"""下面引用Pytorch的Module类的注释
    Base class for all neural network modules.

    Your models should also subclass this class.

    Modules can also contain other Modules, allowing to nest them in
    a tree structure. You can assign the submodules as regular attributes::

        import torch.nn as nn
        import torch.nn.functional as F

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 20, 5)
                self.conv2 = nn.Conv2d(20, 20, 5)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                return F.relu(self.conv2(x))

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call :meth:`to`, etc.
    """

    training: bool #控制traing/testing状态
    _parameters: Dict[str, Optional[Parameter]] # 在训练过程中会随BP而更新的参数
    _buffers: Dict[str, Optional[Tensor]] # 在训练过程中不会随着BP而更新的参数
    _modules: Dict[str, Optionsl['Module']] # 子神经网络模块

    """
    buffers与parameters唯一的区别就是前者不会随BP更新而后者会
    至于为什么要将参数分为buffers和parameters
    而不直接将所有参数都作为parameters，不随BP更新的require_grad设置为False
    这个问题没找到合适的解答，但毫无疑问这样做增强了代码的可读性
    关于buffers与parameters可以参考：
    https://zhuanlan.zhihu.com/p/89442276
    """

    
    def __init__(self) -> None:
        """
        Calls super().__setattr__('a', a) instead of the typical self.a = a
        to avoid Module.__setattr__ overhead. Module's __setattr__ has special
        handling for parameters, submodules, and buffers but simply calls into
        super().__setattr__ for all other attributes.
        """
        super().__setattr__('training', True)
        super().__setattr__('_parameters', OrderedDict())
        super().__setattr__('_buffers', OrderedDict())
        super().__setattr__('_modules', OrderedDict())

    def register_buffer(self, name: str, tensor: Optional[Tensor], persistent: bool = True) -> None:
        r"""Add a buffer to the module.

        This is typically used to register a buffer that should not to be
        considered a model parameter. For example, BatchNorm's ``running_mean``
        is not a parameter, but is part of the module's state. Buffers, by
        default, are persistent and will be saved alongside parameters. This
        behavior can be changed by setting :attr:`persistent` to ``False``. The
        only difference between a persistent buffer and a non-persistent buffer
        is that the latter will not be a part of this module's
        :attr:`state_dict`.

        Buffers can be accessed as attributes using given names.

        Args:
            name (str): buffer的名字
            tensor (Tensor or None): buffer的值. If ``None``, then operations
                that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                the buffer is **not** included in the module's :attr:`state_dict`.
            persistent (bool)：这个buffer是否是这个模型的一部分，决定了在保存模型时是否要保存这个buffer

        Example::

            >>> # xdoctest: +SKIP("undefined vars")
            >>> self.register_buffer('running_mean', torch.zeros(num_features))

        """
        if persistent is False and isinstance(self, torch)

    

    def __call__(self, *x):
        return self.forward(*x)

    def __setattr__(self, __name, __value):
        self.__dict__[__name] = __value
        if isinstance(__value, Parameter):
            self._parameters[__name] = __value
        if isinstance(__value, Module):
            self._modules[__name] = __value
            for key in __value._parameters:
                self._parameters[__name + "." + key] = __value._parameters[key]

    def parameters(self):
        for param in self._parameters.values():
            yield param

    def add_module(self, name, module):
        self._modules[name] = module

    def modules(self):
        for module in self._modules.values():
            yield module

    def train(self, mode=True):
        self.training = mode
        for module in self.modules():
            module.train(mode)

    def forward(self, x):
        raise NotImplementedError

    def eval(self):
        return self.train(False)
