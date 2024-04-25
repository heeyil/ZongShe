# 下面这种方法模型保存的方法是究极简单版
# pytorch的方法太复杂了
# 故等到后面用c++重构底层代码时再修改此篇

import json
import os
import io
import datetime
from typing import Any, BinaryIO, Callable, cast, Dict, Optional, Type, Tuple, Union, IO, List
from typing_extensions import TypeAlias, TypeGuard
from tensor import Graph

# 目前一般用到的都是'str',所以之后的实现默认FILE_LIKE为str类型的，即要保存的文件路径
FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]


def save(obj: object, f: FILE_LIKE,  meta=None,model_file_name='model.json',weights_file_name='weights.npz') -> None:
             
    if not os.path.exists(f):
        os.makedirs(f)
    
    # 元信息，主要记录模型的保存时间和节点值文件名
    meta = {} if meta is None else meta
    meta['save_time'] = str(datetime.datetime.now())
    meta['weights_file_name'] = weights_file_name

    model_json = {'meta': meta}
    graph_json = []
    weights_dict = dict()

    # 开始记录parameters和buffers并保存
    params_buffers = obj.params_and_buffers_saved()
    for k, v in params_buffers：
        node_json = {
            'node_type': k,
            'name': v.name,
        }

        if v.data is not None:
            node_json['dim'] = v.shape

        graph_json.append(node_json)

        # 保存值
        weights_dict[v.name] = v.data

    model_json['graph'] = graph_json

    # json格式保存计算图元信息
    model_file_path = os.path.join(f, model_file_name)
    with open(model_file_path, 'w') as model_file:
        json.dump(model_json, model_file, indent=4)
        print('Save model into file: {}'.format(model_file.name))

    # npz格式保存节点值
    weights_file_path = os.path.join(f, weights_file_name)
    with open(weights_file_path, 'wb') as weights_file:
        np.savez(weights_file, **weights_dict)
        print('Save weights to file: {}'.format(weights_file.name))
