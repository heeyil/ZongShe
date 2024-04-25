# 下面这种方法模型保存的方法就是老师上课讲的方法
# 我一开始并不想用这种方法，因为实现过于简单，且无法保存整个模型(nn.Module)
# 但我想实现的方法太过复杂，有心无力
# 故等到后面用c++重构底层代码时再修改此篇(估计到综设结束都不会去做，因为不会c++)

import json
import os
import io
import datetime
from typing import Any, BinaryIO, Callable, cast, Dict, Optional, Type, Tuple, Union, IO, List
from typing_extensions import TypeAlias, TypeGuard
from tensor import Graph

# 目前一般用到的都是'str',所以之后的实现默认FILE_LIKE为str类型的，即要保存的文件路径
FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]


class Saver:
    r"""
    模型、计算图保存和加载工具类
    模型保存微两个单独文件：
    1.计算图自身的结构元信息
    2.节点的值，变量节点的权值
    """

    def __init__(self, f: FILE_LIKE):
        self.f = f
        if not os.path.exists(self.f):
            os.makedirs(self.f)


    def save(self, graph=None, meta=None, service_signature=None,  model_file_name='model.json',  weights_file_name='weights.npz'):
        """
        把计算图保存到文件中
        """

        # 元信息，记录模型的保存时间和节点值文件名
        meta = {} if meta is None else meta
        meta['save_time'] = str(datatime.datatime.now())
        meta['weights_file_name'] = weights_file_name

        # 服务接口描述
        service = {} if service_signature is None else service_signature

        #保存
        self._save_model_and_weights(
            graph, meta, service, model_file_name, weights_file_name)


    def _save_model_and_weights(self, graph, meta, service, model_file_name, weights_file_name):
        model_json = {
            'meta': meta,
            'service':service
        }
        graph_json = []
        weights_dict = ditc()

        # 把节点元信息保存为dict/json格式
        for node in Graph.node_list:
            node_json = {
                'node_type': node.__class__.__name__,
                'name': node.name,
                'parents': [parent.name for parent in node.parents],
                'children': [child.name for child in node.children],
            }

            # 保存节点的dim信息
            if node.data is not None:
                node_json['dim] = node.shape

            
            
                 
        
    
