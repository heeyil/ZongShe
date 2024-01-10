import numpy as np


class Graph:
    """
    计算图
    """
    node_list: list = list()

    """
    @classmethod 是 Python 中的一个装饰器，用于定义类方法。类方法是与类相关联的方法，而不是与类的实例相关联的方法。
    使用 @classmethod 装饰器可以让 Python 解释器知道这是一个类方法，从而正确地处理它。类方法可以通过类名调用，也可以通过类的实例调用。
    当使用类名调用类方法时，会自动将类作为第一个参数传递给方法。当使用类的实例调用类方法时，同样会自动将类作为第一个参数传递给方法。
    类方法在调用时，第一个参数是类本身，通常将其命名为 cls。这样可以在类方法中访问和修改类的属性，或创建类的实例。
    """

    @classmethod
    def add(cls, node):
        """ 将结点添加进计算图中"""
        cls.node_list.append(node)

    @classmethod
    def clear(cls):
        """list.clear()是列表（list）的一个方法，用于清除列表中的所有元素，使其变为空列表"""
        cls.node_list.clear()

    @classmethod
    def free_graph(cls):
        """
        Pytorch在每次反向传播后，会销毁计算图，实际是删除满足以下条件的计算图中的结点
        1.requires_grad = True
        2.由运算产生的结点

        删除操作包括：
        1.将该节点踢出计算图节点集合；
        2.删除该节点的前驱和后继信息。

        总结来说就是保留requires_grad=True的叶子结点
        """

        new_list = []
        for node in Graph.node_list:
            node.children.clear()
            if node.is_leaf:
                new_list.append(node)

            node.parents.clear()
        Graph.node_list = new_list


class Tensor:
    """
    将NumPy包装成可微分张量
    """

    def __init__(self, data, dtype=None, requires_grad=False):
        self.data = np.array(data, dtype)
        self.requires_grad = requires_grad

        """Only Tensors of floating point dtype can require gradients!"""
        assert not (self.requires_grad and self.dtype != float)

        """
        设置tensor的梯度grad
        如果requires_grad=True则初始化为跟data相同shape的全零array
        否则为None
        """
        self.grad = np.zeros_like(self.data) if self.requires_grad else None

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
    def is_leaf(self):
        # 判断是否为叶节点:需要求导且无上游节点的节点为叶节点
        return self.requires_grad and len(self.parents) == 0

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        return self.transpose()

    def astype(self, new_type):
        # 类型转换，我们不允许可求导节点的类型转换
        assert not self.requires_grad
        self.data.astype(new_type)

    def reshape(self, *new_shape):
        return reshape(self, new_shape)

    def transpose(self, *axes):
        return transpose(self, axes if len(axes) != 0 else None)

    def build_edge(self, node):
        # 构建两节点的有向边
        self.children.append(node)
        node.parents.append(self)

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

    def __len__(self):
        return len(self.data)

    def __pos__(self):
        return 1 * self

    def __neg__(self):
        return -1 * self

    def __abs__(self):
        return abs(self)

    def __getitem__(self, key):
        return get_slice(self, key)

    def backward(self, retain_graph=False):
        """
        retain_graph 是否保留计算图
        """
        if self not in Graph.node_list:
            return

        assert self.data.ndim == 0

        self.grad = np.ones_like(self.data)
        for i in range(len(Graph.node_list) - 1, -1, -1):  # 逆序遍历计算图的结点列表
            if Graph.node_list[i] is self:
                id = i

        for node in Graph.node_list[id::-1]:  # 从列表的id位置开始向前遍历列表，得到的node是计算图中的结点
            grad = node.grad
            for parent in [p for p in node.parents if p.requires_grad]:
                add_grad = node.grad_fn(parent, grad)
                if add_grad.shape != parent.shape:
                    add_grad = np.sum(add_grad,
                                      axis=tuple(-i for i in range(1, parent.ndim + 1)
                                                 if parent.shape[-i] == 1),
                                      keepdims=True)
                    add_grad = np.sum(add_grad,
                                      axis=tuple(range(add_grad.ndim - parent.ndim)),
                                      )
                parent.grad += add_grad

            if not node.is_leaf:
                node.grad = None

        if not retain_graph:
            Graph.free_graph()

    def zero_grad(self):
        self.grad = np.zeros(self.shape)

    def item(self):
        return self.data.item()


class UnaryOperator(Tensor):
    """ 一元操作的基类 """

    def __init__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        Tensor.__init__(self, data=self.forward(x),
                        requires_grad=x.requires_grad)

        if self.requires_grad:
            x.build_edge(self)

    def forward(self, x):
        raise NotImplementedError

    def grad_fn(self, x, grad):
        raise NotImplementedError


class BinaryOperator(Tensor):
    """ 二元操作的基类 """

    def __init__(self, x, y):
        if not isinstance(x, Tensor) and isinstance(y, Tensor):
            x = Tensor(x)
        elif isinstance(x, Tensor) and not isinstance(y, Tensor):
            y = Tensor(y)
        elif not (isinstance(x, Tensor) and isinstance(y, Tensor)):
            x, y = Tensor(x), Tensor(y)

        Tensor.__init__(self, data=self.forward(x, y),
                        requires_grad=(x.requires_grad or y.requires_grad))

        if self.requires_grad:
            x.build_edge(self)
            y.build_edge(self)

    def forward(self, x, y):
        raise NotImplementedError

    def grad_fn(self, x, grad):
        raise NotImplementedError


class add(BinaryOperator):
    def forward(self, x, y):
        return x.data + y.data

    def grad_fn(self, node, grad):
        return grad[...]


class sub(BinaryOperator):
    def forward(self, x, y):
        return x.data - y.data

    def grad_fn(self, node, grad):
        if node is self.parents[0]:
            return grad[...]
        return -grad


class mul(BinaryOperator):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self, x, y):
        return x.data * y.data

    def grad_fn(self, node, grad):
        if node is self.parents[0]:
            return self.parents[1].data * grad
        return self.parents[0].data * grad


class div(BinaryOperator):
    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self, x, y):
        return x.data / y.data

    def grad_fn(self, node, grad):
        temp = grad / self.parents[1].data
        if node is self.parents[0]:
            return temp
        return -self.data * temp


class pow(BinaryOperator):

    def __init__(self, x: Tensor, y: Tensor) -> None:
        super().__init__(x, y)

    def forward(self, x: Tensor, y: Tensor):
        return x.data ** y.data

    def grad_fn(self, node: Tensor, grad: np.ndarray) -> np.ndarray:
        if node is self.parents[0]:
            return (self.data * self.parents[1].data / node.data) * grad
        else:
            return self.data * np.log(self.parents[0].data) * grad


class matmul(BinaryOperator):

    def __init__(self, x, y):
        super().__init__(x, y)

    def forward(self, x, y):
        return x.data @ y.data

    def grad_fn(self, node, grad):
        if node is self.parents[0]:
            if self.parents[1].ndim == 1:
                return np.expand_dims(grad, -1) @ np.expand_dims(
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

    def forward(self, x):
        return np.abs(x)

    def grad_fn(self, x, grad):
        mask = np.zeros(x.shape)
        mask[x > 0] = 1.
        mask[x < 0] = -1.
        return grad * mask


class sum(UnaryOperator):

    def __init__(self, x, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x):
        return np.sum(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x, grad):
        if not (self.axis is None or self.keepdims):
            grad = np.expand_dims(grad, axis=self.axis)
        return np.ones(x.shape) * grad


class mean(UnaryOperator):

    def __init__(self, x, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x):
        return np.mean(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x, grad):
        if not (self.axis is None or self.keepdims):
            grad = np.expand_dims(grad, axis=self.axis)
        return np.ones(x.shape) * grad * self.data.size / x.data.size


class max(UnaryOperator):

    def __init__(self, x, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.max(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x, grad):
        if self.keepdims or self.axis is None:
            full_dim_y = self.data
        else:
            # 还原维度
            full_dim_y = np.expand_dims(self.data, axis=self.axis)
            grad = np.expand_dims(grad, axis=self.axis)
        return (full_dim_y == x.data).astype(float) * grad


class min(UnaryOperator):

    def __init__(self, x, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims
        super().__init__(x)

    def forward(self, x: Tensor) -> np.ndarray:
        return np.min(x.data, axis=self.axis, keepdims=self.keepdims)

    def grad_fn(self, x, grad):
        if self.keepdims or self.axis is None:
            full_dim_y = self.data
        else:
            # 还原维度
            full_dim_y = np.expand_dims(self.data, axis=self.axis)
            grad = np.expand_dims(grad, axis=self.axis)
        return (full_dim_y == x.data).astype(float) * grad


class argmax(Tensor):
    def __init__(self, x, axis=None):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        self.axis = axis
        super().__init__(self.forward(x))

    def forward(self, x: Tensor) -> np.ndarray:
        return np.argmax(x.data, axis=self.axis)


class exp(UnaryOperator):

    def forward(self, x: Tensor):
        return np.exp(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return self.data * grad


class log(UnaryOperator):

    def forward(self, x: Tensor):
        return np.log(x.data)

    def grad_fn(self, x: Tensor, grad: np.ndarray) -> np.ndarray:
        return grad / x.data


class maximum(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return np.maximum(x.data, y.data)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (self.data == x.data) * grad


class minimum(BinaryOperator):
    def forward(self, x: Tensor, y: Tensor) -> np.ndarray:
        return np.minimum(x, y)

    def grad_fn(self, x: Tensor, grad) -> np.ndarray:
        return (self.data == x) * grad


def sqrt(x):
    return x ** 0.5


def square(x):
    return x * x


# 非计算函数
class reshape(UnaryOperator):

    def __init__(self, x, new_shape):
        self.new_shape = new_shape
        super().__init__(x)

    def forward(self, x):
        return x.data.reshape(self.new_shape)

    def grad_fn(self, x, grad):
        return grad.reshape(x.shape)


class transpose(UnaryOperator):

    def __init__(self, x, axes):
        self.axes = axes
        super().__init__(x)

    def forward(self, x):
        return x.data.transpose(self.axes)

    def grad_fn(self, x, grad):
        if self.axes is None:
            return grad.transpose()
        return grad.transpose(tuple(np.argsort(self.axes)))


class concatenate(Tensor):

    def __init__(self, tensors, axis=0):
        requires_grad = False
        self.tensors = tensors
        self.axis = axis
        self.indices = [0]

        for i in range(len(self.tensors)):
            assert isinstance(
                tensors[i],
                Tensor), "Concatenate elements in 'tensors' must be 'Tensor'"

            requires_grad = requires_grad or self.tensors[i].requires_grad
            self.indices.append(self.indices[-1] +
                                self.tensors[i].shape[self.axis])
        super().__init__(self.forward(),
                         requires_grad=requires_grad)
        if self.requires_grad:
            for i in range(len(self.tensors)):
                self.tensors[i].build_edge(self)

    def forward(self):
        return np.concatenate([t.data for t in self.tensors], axis=self.axis)

    def grad_fn(self, x, grad: np.ndarray):
        x_id = self.tensors.index(x)
        start = self.indices[x_id]
        end = self.indices[x_id + 1]
        slc = [slice(None)] * grad.ndim
        slc[self.axis] = slice(start, end)
        return grad[tuple(slc)]


class get_slice(UnaryOperator):
    def __init__(self, x, key):
        if isinstance(key, Tensor):
            self.key = key.data
        else:
            self.key = key
        super().__init__(x)

    def forward(self, x):
        return x.data[self.key]

    def grad_fn(self, x, grad):
        full_grad = np.zeros(x.shape)
        full_grad[self.key] = grad
        return full_grad


def empty(shape, dtype=None, requires_grad=False):
    return Tensor(np.empty(shape),
                  dtype=dtype,
                  requires_grad=requires_grad)