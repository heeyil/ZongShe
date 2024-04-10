from typing import Any
import numpy as np


try:
    import cupy as cp
    cuda_available: bool = True
except ImportError as e:
    cp = object()
    cuda_available: bool = False
    print(e)


def is_available() -> bool:
    return cuda_available


def device_count() -> int:
    if is_available():
        return cp.cuda.runtime.getDeviceCount()
    else:
        return 0


def current_device() -> int:
    return cp.cuda.runtime.getDevice()


def set_device(device: int) -> None:
    return cp.cuda.runtime.setDevice(device)


class Device:
    def __init__(self, device: Any = None) -> None:
        # 对device类型的判断
        if isinstance(device, str):
            if device == 'cpu':
                self.device = 'cpu'
            elif device == 'cuda':
                self.device = 'cuda'
                self.device_id = current_device()
            else:
                assert len(device) > 5 and device[:5] == 'cuda:' and device[5:].isdigit()
                self.device = 'cuda'
                self.device_id = int(device[5:])
        elif isinstance(device, int):
            self.device = 'cuda'
            self.device_id = device
        elif device is None:
            self.device = 'cpu'
        elif isinstance(device, Device):
            self.device = device.device
            if self.device != 'cpu':
                self.device_id = device.device_id
        if self.device == 'cuda':
            self.device = cp.cuda.Device(self.device_id)
        # 最终确认是否初始化成功，保证device在cpu或是gpu
        assert self.device_id == 'cpu' or is_available()

    # 设置该类输出的实例化信息
    def __repr__(self) -> str:
        if self.device == 'cpu':
            return 'Device(type=cpu)'
        else:
            return 'Device(type=cuda, index={})'.format(self.device_id)

    def __eq__(self, device: Any) -> bool:
        assert isinstance(device, Device)
        if self.device == 'cpu':
            return device.device == 'cpu'  # '=='返回的是一个布尔变量
        else:
            if device.device == 'cpu':
                return False
            return self.device == device.device

    @property
    def xp(self):
        """
        cpu : numpy
        gpu : cumpy
        """
        return np if self.device == 'cpu' else cp

    """
    下面是两个上下文管理器的函数
    用于处理异常操作
    详情请浏览相关文章，这里给出两个链接参考
    https://www.cnblogs.com/lipijin/p/4460487.html
    https://zhuanlan.zhihu.com/p/356060929
    """
    def __enter__(self):
        if self.device != 'cpu':
            return self.device.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.device != 'cpu':
            return self.device.__exit__(exc_type, exc_val, exc_tb)


