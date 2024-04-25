import numpy as np
from autograd import is_grad_enable, no_grad
from typing import Any, List, Tuple, Union, Optional
from cuda import Device


class Graph:
    """
    动态计算图
    """
    node_list: list = list()

    """
    @classmethod 是 Python 中的一个装饰器，用于定义类方法。类方法是与类相关联的方法，而不是与类的实例相关联的方法。
    当使用类名调用类方法时，会自动将类作为第一个参数传递给方法。当使用类的实例调用类方法时，同样会自动将类作为第一个参数传递给方法。
    类方法在调用时，第一个参数是类本身，通常将其命名为 cls。这样可以在类方法中访问和修改类的属性，或创建类的实例。
    详情参考以下链接
    https://zhuanlan.zhihu.com/p/35643573
    """

    @classmethod
    def add(cls, node):
        """ 将结点添加进计算图中"""
        cls.node_list.append(node)

    @classmethod
    def clear(cls):
        """清空计算图"""
        """list.clear()是列表（list）的一个方法，用于清除列表中的所有元素，使其变为空列表"""
        cls.node_list.clear()

    @classmethod
    def free_graph(cls):
        """
        释放计算图，但不会删除叶子节点
        """
        """
        Pytorch在每次反向传播后，会销毁计算图，实际是删除满足以下条件的计算图中的结点
        1.requires_grad = True
        2.由运算产生的结点

        删除操作包括：
        1.将该节点踢出计算图节点集合；
        2.删除该节点的前驱和后继信息。

        总结来说就是保留requires_grad=True的叶子结点（现在看起来这句话有点问题，我忘了我为什么写这句话）
        """

        new_list = []
        for node in Graph.node_list:
            node.children.clear()
            if node.is_leaf:
                new_list.append(node)

            node.parents.clear()
        Graph.node_list = new_list


_tensor_count = 0


class Tensor:
    """
    将数据(NumPy数组)包装成可微分张量

    Parameters
    ----------
    data : ndarray
        张量数据，只要是np.array能够转换的数据;
    requires_grad : bool, default=False
        是否需要求梯度;
    dtype : default=None
        数据类型，和numpy数组的dtype等价
    name : Tensor的名称

    Attributes
    ----------
    data : numpy.ndarray
        核心数据，为NumPy数组;
    requires_grad : bool
        是否需要求梯度;
    grad : numpy.ndarray
        梯度数据，为和data相同形状的数组(初始化为全0);
    unique_id : int
        没有为Tensor取名时，用此作为其名;
    children : list[Tensor]
        下游节点列表；
    parents : list[Tensor]
        上游节点列表.
    """

    def __init__(
            self,
            data: Any,
            dtype=None,
            device: Union[Device, int, str, None] = None,
            name: Optional[str] = None,
            requires_grad: bool = False
    ) -> None:

        global _tensor_count
        _tensor_count += 1
        self.unique_id = _tensor_count
        if name is not None:
            self.name = name
        else:
            self.name = str(self.unique_id)
        
        if isinstance(data, Tensor):
            data = data.data

        if not isinstance(device, Device):
            device = Device(device)

        self.device: Device = device
        # 为什么下面用with
        with self.device:
            """
            device = cpu, 用numpy
            device = gpu, 用cumpy
            """
            self.data = self.xp.array(data, dtype)

        # 对于一个tensor的是否需要梯度，还需要设置的此时整个代码区是否需要求梯度，即"with no_grad:"or"with enable_grad:"
        self.requires_grad: bool = requires_grad and is_grad_enable()

        """只有float的数据才能求梯度,um3?"""
        assert not (
                self.requires_grad and self.dtype != float
        ), "Only Tensors of floating point dtype can require gradients!"

        """
        设置tensor的梯度grad
        如果requires_grad=True则初始化为跟data相同shape的全零array
        否则为None
        """
        self.grad = self.xp.zeros_like(self.data) if self.requires_grad else None

        # 初始化子节点和父节点列表
        self.children = list()
        self.parents = list()

        if self.requires_grad:
            # 在动态计算图中不需要求梯度的结点不出现在计算图中
            Graph.add(self)

    """
    @property 是 Python 中的一个装饰器，主要用于将一个方法变成属性调用。
    这使得我们可以在不调用方法的情况下访问其返回的值，就像访问属性一样。
    
    这一点在pytorch中有体现
    """

    @property
    def is_leaf(self) -> bool:
        # 判断是否为叶节点:需要求导且无上游节点的节点为叶节点
        return self.requires_grad and len(self.parents) == 0

    @property
    def shape(self) -> Tuple[int]:
        """张量的形状，用法同NumPy.

        Example
        -------
        > Tensor([[2, 2]]).shape
        (1, 2)
        """
        return self.data.shape

    @property
    def ndim(self):
        """张量的维度，用法同NumPy.

        Example
        -------
        > Tensor([[2, 2]]).ndim
        2
        """
        return self.data.ndim

    @property
    def dtype(self):
        """张量的数据类型，用法同NumPy.

        Example
        -------
        > Tensor([[2, 2]]).dtype
        dtype('int64')
        """
        return self.data.dtype

    @property
    def size(self):
        """张量的元素个数，用法同NumPy.

        Example
        -------
        >Tensor([[1, 1]]).size
        2
        """
        return self.data.size

    @property
    def T(self):
        """张量的转置"""
        return self.transpose()

    def astype(self, new_type):
        # 类型转换，不允许可求导节点的类型转换
        assert not self.requires_grad
        self.data.astype(new_type)

    def reshape(self, *new_shape):
        return reshape(self, new_shape)

    def transpose(self, *axes):
        return transpose(self, axes if len(axes) != 0 else None)

    def max(
        self,
        axis: Union[int, Tuple, None] = None,
        keepdims: bool = False,
    ):
        return max(self, axis, keepdims)

    def min(
        self,
        axis: Union[int, Tuple, None] = None,
        keepdims: bool = False,
    ):
        return min(self, axis, keepdims)

    def sum(
            self,
            axis: Union[int, Tuple, None] = None,
            keepdims: bool = False,
    ):
        return sum(self, axis, keepdims)

    def build_edge(self, node):
        # 构建两节点的有向边
        self.children.append(node)
        node.parents.append(self)

    """
    下面的都是用魔法方法实现运算符重载
    简单地说，就是使非常数的那种操作数能像常数那样进行运算
    比如说，对两个tensor a和b，肯定不能直接像1+2那样运算
    如果想要实现a+b这样直接算的话，就需要使用魔法方法进行运算符重载
    比如说加法重载，就要在tensor类中定义重载加法：def __add__():
    然后在函数里面写代码具体实现这个加法
    最后，就可以实现两个自定义类a+b这样直接算
    u闹3?
    
    下面这些肯定是不够的，还需要日后补充。
    """

    def __add__(self, x):
        return add(self, x)

    def __radd__(self, x):
        return add(x, self)

    def __sub__(self, x):
        return sub(self, x)

    def __rsub__(self, x):
        return sub(x, self)

    def __mul__(self, x):
        return mul(self, x)

    def __rmul__(self, x):
        return mul(x, self)

    def __matmul__(self, x):
        return matmul(self, x)

    def __rmatmul__(self, x):
        return matmul(x, self)

    def __truediv__(self, x):
        return div(self, x)

    def __rtruediv__(self, x):
        return div(x, self)

    def __pow__(self, x):
        return pow(self, x)

    def __rpow__(self, x):
        return pow(x, self)

    def __len__(self) -> int:
        return len(self.data)

    def __pos__(self):
        return 1 * self

    def __neg__(self):
        return -1 * self

    def __abs__(self):
        return abs(self)

    def __getitem__(self, key):
        return get_slice(self, key)

    def backward(self, retain_graph: bool = False):
        """
        retain_graph 是否保留计算图
        """
        if self not in Graph.node_list:
            return

        assert self.data.ndim == 0

        self.grad = self.xp.ones_like(self.data)
        for i in range(len(Graph.node_list) - 1, -1, -1):  # 逆序遍历计算图的结点列表
            if Graph.node_list[i] is self:
                id = i
                break

        for node in Graph.node_list[id::-1]:  # 从列表的id位置开始向前遍历列表，得到的node是计算图中的结点
            grad = node.grad
            for parent in [p for p in node.parents if p.requires_grad]:
                add_grad = node.grad_fn(parent, grad)
                if add_grad.shape != parent.shape:
                    add_grad = self.xp.sum(
                        add_grad,
                        axis=tuple(-i for i in range(1, parent.ndim + 1)
                                   if parent.shape[-i] == 1),
                        keepdims=True
                    )
                    add_grad = self.xp.sum(
                        add_grad,
                        axis=tuple(range(add_grad.ndim - parent.ndim)),
                    )
                parent.grad += add_grad

            if not node.is_leaf:
                node.grad = None

        if not retain_graph:
            Graph.free_graph()

    def zero_grad(self):
        """梯度归零"""
        self.grad = self.xp.zeros(self.shape)

    def numpy(self) -> np.ndarray:
        """返回Tensor的内部数据，即Numpy数组（拷贝）"""
        return self.cpu().data.copy()

    def item(self):
        return self.data.item()

    def to(self, device):
        device = Device(device)
        if self.device == device:
            return self
        elif device.device == "cpu":  # cuda -> cpu
            return Tensor(self.data.get(), dtype=self.dtype, device=device)
        else:  # cpu -> cuda
            return Tensor(self.data, dtype=self.dtype, device=device)

    def cpu(self):
        return self.to("cpu")

    def cuda(self):
        return self.to("cuda")

    @property
    def xp(self):
        return self.device.xp


"""
下面是定义Tensor的基本运算算子
"""


class UnaryOperator(Tensor):
    """ 一元操作的基类 """

    def __init__(self, x: Tensor) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.device = x.device
        super().__init__(
            data=self.forward(x),
            device=x.device,
            requires_grad=is_grad_enable() and x.requires_grad,
        )

        if self.requires_grad:
            x.build_edge(self)

    def forward(self, x: Tensor) -> np.ndarray:
        """前向传播函数，即定义该算子的具体运算操作"""
        raise NotImplementedError

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        """
        反向传播函数
        x : Tensor
            下游节点
        grad : ndarray
            上游流入该节点的梯度
        """
        raise NotImplementedError

    # 定义该类实例化信息
    def __repr__(self) -> str:
        return "Tensor({}, op={})".format(self.data, self.__class__.__name__)


class BinaryOperator(Tensor):
    """ 二元操作的基类 """

    def __init__(self, x: Tensor, y: Tensor) -> None:
        if not isinstance(x, Tensor) and isinstance(y, Tensor):
            x = Tensor(x, device=y.device)
        elif isinstance(x, Tensor) and not isinstance(y, Tensor):
            y = Tensor(y, device=x.device)
        elif not (isinstance(x, Tensor) and isinstance(y, Tensor)):
            x, y = Tensor(x), Tensor(y)

        assert x.device == y.device

        super().__init__(
            data=self.forward(x, y),
            device=x.device,
            requires_grad=is_grad_enable
            and (x.requires_grad or y.requires_grad),
        )

        if self.requires_grad:
            x.build_edge(self)
            y.build_edge(self)

    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        raise NotImplementedError

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "Tensor({}, op={})".format(self.data, self.__class__.__name__)


"""
下面就是对之前运算符重载时没实现的具体运算操作应用一元/二元算子类的思想
"""


class add(BinaryOperator):
    """
    加法算子

    Example
    -------
    > x = Tensor(1.)
    > y = Tensor(2.)
    > z = add(x, y) # 在Tensor类中进行了重载，所以也可以写成
    > z = x + y
    """
    def forward(self, x: Tensor, y: Tensor):
        return x.data + y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        return grad[...]


class sub(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor):
        return x.data - y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        if node is self.parents[0]:
            return grad[...]
        return -grad


class mul(BinaryOperator):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data * y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        if node is self.parents[0]:
            return self.parents[1].data * grad
        return self.parents[0].data * grad


class div(BinaryOperator):
    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data / y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray):
        temp = grad / self.parents[1].data
        if node is self.parents[0]:
            return temp
        return -self.data * temp


class pow(BinaryOperator):
    """
    幂运算算子
    """

    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data ** y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray) -> np.ndarray:
        if node is self.parents[0]:
            return (self.data * self.parents[1].data / node.data) * grad
        else:
            return self.data * self.xp.log(self.parents[0].data) * grad


class matmul(BinaryOperator):
    """
    矩阵乘法算子
    """

    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return x.data @ y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray) -> np.ndarray:
        if node is self.parents[0]:
            if self.parents[1].ndim == 1:
                return self.xp.expand_dims(grad, -1) @ np.expand_dims(
                    self.parents[1].data, -2)
            elif self.parents[1].ndim > 2:
                shape = list(range(self.parents[1].ndim))
                shape[-1], shape[-2] = shape[-2], shape[-1]
                return grad @ self.parents[1].data.transpose(*shape)
            return grad @ self.parents[1].data.T
        else:
            if self.parents[0].ndim == 1:
                return np.expand_dims(self.parents[0].data, -1) @ np.expand_dims(grad, -2)
            elif self.parents[0].ndim > 2:
                shape = list(range(self.parents[0].ndim))
                shape[-1], shape[-2] = shape[-2], shape[-1]
                return self.parents[0].data.transpose(*shape) @ grad
            return self.parents[0].data.T @ grad


class abs(UnaryOperator):
    """
    绝对值算子
    """

    def forward(self, x: Tensor) -> np.ndarray:
        return self.xp.abs(x)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        mask = self.xp.zeros(x.shape)
        mask[x > 0] = 1.
        mask[x < 0] = -1.
        return grad * mask


class sum(UnaryOperator):
    """
    求和算子，在Tensor类中扩展为类方法

    Parameters
    ----------
    axis : None
        求和方向(轴)
    keepdims : bool, default=False
        是否保留原来维度

    Example
    -------
    > x = Tensor(
            [[1, 2, 3],
            [4, 5, 6]]
        )
    > s1 = x.sum(0) # [5, 7, 9]
    > s2 = x.sum(1) # [6, 15]
    > s3 = sum(x, keepdims=True) # [[21]]
    ```
    """

    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return self.xp.sum(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if not (self.axis is None or self.keepdims):
            grad = self.xp.expand_dims(grad, axis=self.axis)
        return self.xp.ones(x.shape) * grad


class mean(UnaryOperator):
    """
    求均值算子
    """

    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return self.xp.mean(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if not (self.axis is None or self.keepdims):
            grad = self.xp.expand_dims(grad, axis=self.axis)
        return self.xp.ones(x.shape) * grad * self.data.size / x.data.size


class max(UnaryOperator):

    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return self.xp.max(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.keepdims or self.axis is None:
            full_dim_y = self.data
        else:
            # 还原维度
            full_dim_y = self.xp.expand_dims(self.data, axis=self.axis)
            grad = self.xp.expand_dims(grad, axis=self.axis)
        return (full_dim_y == x.data).astype(float) * grad


class min(UnaryOperator):

    def __init__(self, x: Tensor, axis=None, keepdims=False) -> None:
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return self.xp.min(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.keepdims or self.axis is None:
            full_dim_y = self.data
        else:
            # 还原维度
            full_dim_y = self.xp.expand_dims(self.data, axis=self.axis)
            grad = self.xp.expand_dims(grad, axis=self.axis)
        return (full_dim_y == x.data).astype(float) * grad


class argmax(Tensor):
    def __init__(self, x: Tensor, axis=None) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.axis = axis
        self.device = x.device
        super().__init__(self.forward(x), device=self.device)

    def forward(self, x: Tensor) -> np.ndarray:
        return self.xp.argmax(x.data, axis=self.axis)


class argmin(Tensor):
    def __init__(self, x: Tensor, axis=None) -> None:
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.axis = axis
        self.device = x.device
        super().__init__(self.forward(x), device=self.device)

    def forward(self, x: Tensor) -> np.ndarray:
        return self.xp.argmin(x.data, axis=self.axis)


class exp(UnaryOperator):
    """指数运算

    Example
    -------
    > x = Tensor(1.)
    > y = exp(x)
    """

    def forward(self, x: Tensor):
        return self.xp.exp(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return self.data * grad


class log(UnaryOperator):
    """对数运算

    Example
    -------
    > x = Tensor(1.)
    > y = log(x)
    """

    def forward(self, x: Tensor):
        return self.xp.log(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad / x.data


class maximum(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return self.xp.maximum(x.data, y.data)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (self.data == x.data) * grad


class minimum(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return self.xp.minimum(x, y)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (self.data == x) * grad


def sqrt(x: Tensor):
    """平方根函数"""
    return x ** 0.5


def square(x: Tensor):
    """平方函数"""
    return x * x


# 非计算函数
class reshape(UnaryOperator):
    """
    张量形状变换算子

    Parameters
    ----------
    new_shape : tuple
        变换后的形状，用法同NumPy
    """
    def __init__(self, x: Tensor, new_shape: tuple) -> None:
        self.new_shape = new_shape
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return x.data.reshape(self.new_shape)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad.reshape(x.shape)


class transpose(UnaryOperator):
    """
    张量转置算子

    Parameters
    ----------
    axes : tuple
        转置的轴变换，用法同NumPy
    """
    def __init__(self, x: Tensor, axes: tuple = None) -> None:
        self.axes = axes
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return x.data.transpose(self.axes)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        if self.axes is None:
            return grad.transpose()
        return grad.transpose(tuple(np.argsort(self.axes)))


class concatenate(Tensor):
    """对多个张量进行连接，用法类似于`numpy.concatenate`

    Parameters
    ----------
    tensors :
        待连接的张量：
    axis : default=0
        连接轴，默认是沿着第一个轴拼接.
    """

    def __init__(self, tensors: List[Tensor], axis=0) -> None:
        requires_grad = False
        self.tensors = tensors
        self.axis = axis
        self.indices = [0]

        for i in range(len(self.tensors)):
            assert isinstance(
                tensors[i],
                Tensor), "Concatenate elements in 'tensors' must be 'Tensor'"
            if i == 0:
                device = tensors[i].device
            else:
                assert tensors[i].device == device
            requires_grad = requires_grad or self.tensors[i].requires_grad
            self.indices.append(self.indices[-1] +
                                self.tensors[i].shape[self.axis])
        self.device = device
        super().__init__(self.forward(),
                         requires_grad=requires_grad and is_grad_enable(),
                         device=device)
        if self.requires_grad:
            for i in range(len(self.tensors)):
                self.tensors[i].build_edge(self)

    def forward(self):
        return self.xp.concatenate([t.data for t in self.tensors],
                                   axis=self.axis)

    def grad_fn(self, x, grad: np.ndarray):
        x_id = self.tensors.index(x)
        start = self.indices[x_id]
        end = self.indices[x_id + 1]
        slc = [slice(None)] * grad.ndim
        slc[self.axis] = slice(start, end)
        return grad[tuple(slc)]


class get_slice(UnaryOperator):
    """
    切片算子，为Tensor类提供索引和切片接口

    Example
    -------
    > x = Tensor(
            np.arange(12).reshape(3, 4).astype(float),
            requires_grad=True,
        )
    > y = x[:2, :2].sum()
    > y.backward()
    > x.grad
    [[1. 1. 0. 0.]
     [1. 1. 0. 0.]
     [0. 0. 0. 0.]]
    """
    def __init__(self, x: Tensor, key) -> None:
        if isinstance(key, Tensor):
            self.key = key.data
        else:
            self.key = key
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return x.data[self.key]

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        full_grad = self.xp.zeros(x.shape)
        full_grad[self.key] = grad
        return full_grad


def unsqueeze(x: Tensor, axis: Any):
    """等价于numpy的expand_dims"""
    if type(axis) not in (tuple, list):
        axis = (axis, )

    out_ndim = len(axis) + x.ndim
    axis = normalize_axis_tuple(axis, out_ndim)

    shape_it = iter(x.shape)
    shape = [1 if ax in axis else next(shape_it) for ax in range(out_ndim)]
    return x.reshape(*shape)


def normalize_axis_tuple(axis, ndim):
    if axis is None:
        axis = tuple(range(ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    else:
        axis = tuple(axis)
    axis = tuple(np.arange(ndim)[np.asarray(axis) % ndim])
    return axis


def empty(shape, dtype=None, device=None, requires_grad=False):
    return Tensor(np.empty(shape),
                  dtype=dtype,
                  device=device,
                  requires_grad=requires_grad)


def from_numpy(ndarray):
    return Tensor(ndarray)
