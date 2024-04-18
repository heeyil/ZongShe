from collections import OrderedDict
from nn.parameter import Parameter
from cuda import Device, current_device
from autograd import set_grad_enabled
from tensor import Tensor

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
    _non_persistent_buffers_set: Set[str] # 不保存在模型中的buffer放到这里面
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
        super().__setattr__('_non_persistent_buffers_set', set())
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

        """对buffer进行一些简单的检查"""
        if '_buffers' not in self.__dict__:
            raise AttributeError(
                "cannot assign buffer before Module.__init__() call") # 不能在模型初始化前注册buffer
        elif not isinstance(name, str):
            raise TypeError(f"buffer name should be a string")
        elif '.' in name:
            raise KeyError("buffer name can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError(f"attribute '{name}' already exists")
        elif tensor is not None and not isinstance(tensor, Tensor):
            raise TypeError(f"cannot assign this to buffer '{name}'"
                           "(torch Tensor or None requored)"
                           )
        else:
            self._buffers[name] = tensor
            if persistent:
                self._non_persistent_buffers_set.discard(name)
            else:
                self._non_persistent_buffers_set.add(name)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        r"""Add a parameter to the module.
        没有persistent这个参数是因为凡是为parameter的参数都是必定要保存在模型中的
        """
        if '_parameters' not in self.__dict__:
                        raise AttributeError(
                "cannot assign parameter before Module.__init__() call")

        elif not isinstance(name, str):
            raise TypeError(f"parameter name should be a string")
        elif '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError(f"attribute '{name}' already exists")
            
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param, Parameter):
            raise TypeError(f"cannot assign this param to parameter '{name}' "
                            "(torch.nn.Parameter or None required)"
                            )
        """attention!!!!!!!!!!!!!!!!!!!!!!!!!"""
        elif param.grad_fn:
            raise ValueError(
                f"Cannot assign non-leaf Tensor to parameter '{name}'. Model "
                f"parameters must be created explicitly. To express '{name}' "
                "as a function of another Tensor, compute the value in "
                "the forward() method.")
        else:
            self._parameters[name] = param

    def add_module(self, name: str, module: Optional['Module']) -> None:
        r"""Add a child module to the current module.
        将子模型添加到当前模型

        Args:
             name(str): 子模型的名字
             module (Module): 父模型
        """
        if not isinstance(module, Module) and module is not None:
            raise TypeError(f"{this module is not a Module subclass")
        elif not isinstance(name, str):
            raise TypeError(f"module name should be a string")
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError(f"attribute '{name}' already exists")
        elif '.' in name:
            raise KeyError(f"module name can't contain \".\", got: {name}")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module

    def register_module(self, target: str) -> "Module":
        """与add_module功能相同"""
        self.add_module(name, module)

    def get_submodule(self, target: str) -> "Module":
        """Return the submodule given by ``target`` if it exists, otherwise throw an error.
        通俗来说，就是通过查询现有模型中是否有子模型target
        
        下面举例说明

        设模型"A"，其结构如下:
        A(
           (net_b): Module(
               (net_c): Module(
                   (conv):Conv2d(16, 33, kernel_size=(3, 3), stride=(2,2))
               )
               (linear): Linear(in_features=100, out_features=200, bias=True)
           )
        )
        "A"有一个嵌套的子模型"net_b"，"net_b"又有两个子模型"net_c"和"linear"."net_c"又有一个子模型"conv"

        要检查"A"中是否有"linear"这个子模型，可以使用函数
        ``get_submodule("net_b.linear")``
        要检查是否有"conv"这个子模型，使用函数
        ``get_submodule("net_b.net_c.conv")``
        """
        if target == "":
            return self

        """
        对atoms，设有target=a.b.c
        则atoms=[a,b,c]
        """
        atoms: List[str] = target.split(".")
        # mod就是self，是该模型本身
        mod: Module = self

        for item in atoms:

            if not hasattr(mod, item):
                raise AttributeError(mod._get_name() + " has no "
                                     "attribute `" + item + "`")

            """
            使用'getattr'返回当前item的属性值
            """
            mod = getattr(mod, item)

            if not isinstance(mod, Module):
                raise AttributeError("`" + item + "` is not "
                                     "an nn.Module")

        return mod

    def get_parameter(self, target: str) -> "Parameter":
        """Return the parameter given by ``target`` if it exists, otherwise throw an error.
        """
        module_path, _, param_name = target.rpartition(".")

        mod: Module = self.get_submodule(module_path)

        if not hasattr(mod, param_name):
            raise AttributeError(mod._get_name() + " has no attribute `"
                                 + param_name + "`")

        param: Parameter = getattr(mod, param_name)

        if not isinstance(param, Parameter):
            raise AttributeError("`" + param_name + "` is not an "
                                 "nn.Parameter")

        return param

    def get_buffer(self, target: str) -> "Tensor":
        """Return the buffer given by ``target`` if it exists, otherwise throw an error.
        """
        module_path, _, buffer_name = target.rpartition(".")

        mod: Module = self.get_submodule(module_path)

        if not hasattr(mod, buffer_name):
            raise AttributeError(mod._get_name() + " has no attribute `"
                                 + buffer_name + "`")

        buffer: Tensor = getattr(mod, buffer_name)

        if buffer_name not in mod._buffers:
            raise AttributeError("`" + buffer_name + "` is not a buffer")

        return buffer

    def apply(self: T, fn: Callable[['Module'], None]) -> T:
        r"""Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.
        对模型的所有子模型应用函数'fn'

        Typical use includes initializing the parameters of a model.
        最常用的应用是初始化模型中的所有参数
        
        Args:
            fn (:class:`Module` -> None): function to be applied to each submodule

        Returns:
            Module: self

        Example::

            >>> @torch.no_grad()
            >>> def init_weights(m):
            >>>     print(m)
            >>>     if type(m) == nn.Linear:
            >>>         m.weight.fill_(1.0)
            >>>         print(m.weight)
            >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
            >>> net.apply(init_weights)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Linear(in_features=2, out_features=2, bias=True)
            Parameter containing:
            tensor([[1., 1.],
                    [1., 1.]], requires_grad=True)
            Sequential(
              (0): Linear(in_features=2, out_features=2, bias=True)
              (1): Linear(in_features=2, out_features=2, bias=True)
            )

        """
        for module in self.children():
            module.apply(fn)
        fn(self)
        return self

    """
    下面实现将模型转移到GPU上的方法
    关于@overload参考
    https://zhuanlan.zhihu.com/p/489767633
    """
    @overload
    def to(self, device: Optional[DeviceLikeType] = ..., dtype: Optional[dtype] = ...,
           non_blocking: bool = ...) -> Self:
        ...

    @overload
    def to(self, dtype: dtype, non_blocking: bool = ...) -> Self:
        ...

    @overload
    def to(self, tensor: Tensor, non_blocking: bool = ...) -> Self:
        ...

    def to(self, *args, **kwargs):
        r"""Move and/or cast the parameters and buffers.

        This can be called as

        .. function:: to(device=None, dtype=None, non_blocking=False)
           :noindex:

        .. function:: to(dtype, non_blocking=False)
           :noindex:

        .. function:: to(tensor, non_blocking=False)
           :noindex:

        .. function:: to(memory_format=torch.channels_last)
           :noindex:

        Its signature is similar to :meth:`torch.Tensor.to`, but only accepts
        floating point or complex :attr:`dtype`\ s. In addition, this method will
        only cast the floating point or complex parameters and buffers to :attr:`dtype`
        (if given). The integral parameters and buffers will be moved
        :attr:`device`, if that is given, but with dtypes unchanged. When
        :attr:`non_blocking` is set, it tries to convert/move asynchronously
        with respect to the host if possible, e.g., moving CPU Tensors with
        pinned memory to CUDA devices.

        See below for examples.

        .. note::
            This method modifies the module in-place.

        Args:
            device (:class:`torch.device`): the desired device of the parameters
                and buffers in this module
            dtype (:class:`torch.dtype`): the desired floating point or complex dtype of
                the parameters and buffers in this module
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all parameters and buffers in this module
            memory_format (:class:`torch.memory_format`): the desired memory
                format for 4D parameters and buffers in this module (keyword
                only argument)

        Returns:
            Module: self

        Examples::

            >>> # xdoctest: +IGNORE_WANT("non-deterministic")
            >>> linear = nn.Linear(2, 2)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]])
            >>> linear.to(torch.double)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1913, -0.3420],
                    [-0.5113, -0.2325]], dtype=torch.float64)
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
            >>> gpu1 = torch.device("cuda:1")
            >>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
            >>> cpu = torch.device("cpu")
            >>> linear.to(cpu)
            Linear(in_features=2, out_features=2, bias=True)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.1914, -0.3420],
                    [-0.5112, -0.2324]], dtype=torch.float16)

            >>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
            >>> linear.weight
            Parameter containing:
            tensor([[ 0.3741+0.j,  0.2382+0.j],
                    [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
            >>> linear(torch.ones(3, 2, dtype=torch.cdouble))
            tensor([[0.6122+0.j, 0.1150+0.j],
                    [0.6122+0.j, 0.1150+0.j],
                    [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)

        """
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if dtype is not None:
            if not (dtype.is_floating_point or dtype.is_complex):
                raise TypeError('nn.Module.to only accepts floating point or complex '
                                f'dtypes, but got desired dtype={dtype}')
            if dtype.is_complex:
                warnings.warn(
                    "Complex modules are a new feature under active development whose design may change, "
                    "and some modules might not work as expected when using complex tensors as parameters or buffers. "
                    "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
                    "if a complex module does not work as expected.")

        def convert(t):
            try:
                if convert_to_format is not None and t.dim() in (4, 5):
                    return t.to(
                        device,
                        dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking,
                        memory_format=convert_to_format,
                    )
                return t.to(
                    device,
                    dtype if t.is_floating_point() or t.is_complex() else None,
                    non_blocking,
                )
            except NotImplementedError as e:
                if str(e) == "Cannot copy out of meta tensor; no data!":
                    raise NotImplementedError(
                        f"{e} Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() "
                        f"when moving module from meta to a different device."
                    ) from None
                else:
                    raise

        return self._apply(convert)

    # __getattr__用于访问类中是否有'name'代表的属性值
    # 有则返回，无则报错
    def __getattr__(self, name: str) -> Any:
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    """
    ！！！attention please ！！！
    下面是Module类中一个究极重要的方法
    其重要程度就像原神、星铁、火影、睡觉、大吼大叫对牛折翼的重要程度
    没有这些的话牛折翼就不是一个完整的牛折翼，其就不能正常运作
    对Module类也是如此
    通过上面这个例子你应该能明白下面这个'setattr'类有多重要了

    '__setattr__'的作用就是正确地将parameter,buffer,module添加到Module中
    """
    
    """
    '__setattr__'方法语法本身比较简单，该方法常与'__dict__'联系在一起
    关于'__setattr__'参考：
    https://zhuanlan.zhihu.com/p/101004827?from_voters_page=true
    关于'__dict__'参考：
    https://c.biancheng.net/view/2374.html
    关于'niuzheyi'参考：
    https://www.zhihu.com/question/355555256
    https://www.zhihu.com/question/527375088
    """
    
    def __setattr__(self, name: str, value: Union[Tensor, 'Module']) -> None:
        """
        下面这个方法'remove_from'
        实现 删除 '*dice_or_sets'中一堆列表或元组中key为'name'的元素
        这在为parameter,buffer,module赋值时使用
        举个例子：
        假设要将'name'为'weight'的类型为'parameter'的参数添加进模型
        那么'weight'这个名字的参数就只能为parameter，不能是buffer,module
        所以要删除buffer,module中的'name'为'weight'的参数

        这说明 self.dict，self._buffers，self._parameters，self._modules 中的属性应该是互斥的
        """
        def remove_from(*dicts_or_sets):
            for d in dicts_or_sets:
                if name in d:
                    if isinstance(d, dict):
                        del d[name]
                    else:
                        d.discard(name)

        # __dict__本质上就是一个普通的字典

        # 对parameter
        params = self.__dict__.get('_parameters')
        if isinstance(value, Parameter):
            # 下面这段代码检查了继承 nn.Module 的自定义模块是否有正确地初始化父类 nn.Module
            # 这也说明了 super().__init__() 的必要性
            if params is None:
                raise AttributeError(
                    "cannot assign parameters before Module.__init__() call")
            remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
            self.register_parameter(name, value)
        # value并不是Parameter实例，但该name代表的参数已在模型中存在，说明这个value的赋值是修改之前该参数的赋值
        elif params is not None and name in params:
            if value is not None:
                raise TypeError(f"cannot assign this value as parameter '{name}' "
                                "(Parameter or None expected)"
                                )
            self.register_parameter(name, value)
        else:
            # 对module
            modules = self.__dict__.get('_modules')
            if isinstance(value, Module):
                if modules is None:
                    raise AttributeError(
                        "cannot assign module before Module.__init__() call")
                remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
                modules[name] = value
            elif modules is not None and name in modules:
                if value is not None:
                    raise TypeError(f"cannot assign this as child module '{name}' "
                                    "(nn.Module or None expected)"
                                    )
                modules[name] = value
            else:
                # 对buffer
                # 注意到buffer与parameter与module不同
                # buffer没有'if isinstance(value, Tensor):...'
                # 这说明想要给Module增加buffer,self.register_buffer是唯一的方式
                # __setattr__ 只能将 self._buffers 中已有的buffer重新赋值为None或者tensor
                # 目前没搞懂为什么要这样 :(
                
                buffers = self.__dict__.get('_buffers')
                if buffers is not None and name in buffers:
                    if value is not None and not isinstance(value, torch.Tensor):
                        raise TypeError(f"cannot assign this as buffer '{name}' "
                                        "(Tensor or None expected)"
                                        )
                    buffers[name] = value
                else:
                    super().__setattr__(name, value)

    # 在每次删除操作时会被调用
    def __delattr__(self, name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
            self._non_persistent_buffers_set.discard(name)
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        r"""Return an iterator over module parmeters.
        
        经常在使用优化器更新模型参数时使用
        """
        for name, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self,
        prefix: str = '',
        recurse: bool = True,
        remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        r"""Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

        Args:
            prefix (str): prefix to prepend to all parameter names.
            recurse (bool): if True, then yields parameters of this module
                and all submodules. Otherwise, yields only parameters that
                are direct members of this module.
            remove_duplicate (bool, optional): whether to remove the duplicated
                parameters in the result. Defaults to True.

        Yields:
            (str, Parameter): Tuple containing the name and parameter

        Example::

            >>> for name, param in self.named_parameters():
            >>>     if name in ['bias']:
            >>>         print(param.size())

        """
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
        yield from gen
    
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
