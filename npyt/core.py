import os
from typing import Literal, Optional

import numpy as np

from npyt.format import get_end, get_file_ctx, save, load, resize

__all__ = [
    "NPYT",
]


class NPYT:
    """
    带尾巴的NPY格式文件,后面带小尾巴，为数据开始和结束位置

    Notes
    -----
    很多操作没有条件判断，使用时需要注意。

    """

    def __init__(self, filename: str, max_length: int = 1, mode: Literal["r", "r+", "w+"] = "r"):
        """初始化

        Parameters
        ----------
        filename:str
            文件名
        max_length:int
            最大记录长度
        mode:str
            打开模式。

            - r:只读
            - r+:读写
            - w+:新建文件并读写。如果文件存在，则会被清空

        """
        self._filename = filename
        self._max_length = max_length
        self._mode = mode
        self._arr: Optional[np.ndarray] = None
        self._tail: Optional[np.ndarray] = None

    def save(self, array: np.ndarray, end: Optional[int] = None) -> "NPYT":
        """创建文件

        Parameters
        ----------
        array:
            初始数据
        end:int
            结束位置。0表示只创建文件，数据区为空。None表示使用array的长度。

        """
        if "w" in self._mode:
            # 创建文件
            self._max_length = save(get_file_ctx(self._filename, mode="wb+"), array, self._max_length, end)
        elif "+" in self._mode:
            if not os.path.exists(self._filename):
                self._max_length = save(get_file_ctx(self._filename, mode="wb+"), array, self._max_length, end)

        return self

    def load(self) -> "NPYT":
        """加载文件。为以后操作做准备"""
        mode = self._mode.replace('w', 'r')
        self._arr, self._tail = load(self._filename, mode=mode)
        return self

    def resize(self, length: Optional[int] = None) -> "NPYT":
        """文件截断或扩充

        Parameters
        ----------
        length:int
            新的长度。

            - None: 表示截断到有效数据长度
            - 小于原长度: 表示截断到指定长度
            - 大于原长度: 表示扩充到指定长度

        Notes
        -----
        独占时才能使用

        """
        # 一些基本信息
        end = int(self._tail[1])
        length = get_end(end, length)
        arr = self._arr[:1].copy()

        # 释放文件占用
        self._arr = None
        self._tail = None

        # 释放后就可以动文件了
        resize(self._filename, arr, end, length)

        return self

    def append(self, array: np.ndarray) -> "NPYT":
        """插入数据

        Parameters
        ----------
        array:
            插入的数据。

        Notes
        -----
        插入一行数据时要注意，shape为(1, n)，而不是(n,)。否则会报错。
            - arr[0:1] 正确
            - arr[0] 错误

        Raises
        ------
        ValueError
            插入的数据长度超过了文件长度

        """
        start = int(self._tail[1])
        end = start + array.shape[0]
        # TODO 是大文件模式，还是ringbuffer模式呢?
        self._arr[start:end] = array
        self._tail[1] = end

        return self

    def array(self) -> np.ndarray:
        """取原始数组"""
        return self._arr

    def info(self):
        """获取尾巴关键信息"""
        return tuple(self._tail.tolist())

    def clear(self):
        """清空数据。重置位置指针"""
        self._tail[0:2] = 0

    def data(self) -> np.ndarray:
        """取数据区"""
        start = self._tail[0]
        end = self._tail[1]
        return self._arr[start:end]

    def head(self, n: int = 5) -> np.ndarray:
        """取头部数据"""
        start = int(self._tail[0])
        end = start + n
        return self._arr[start:end]

    def tail(self, n: int = 5) -> np.ndarray:
        """取尾部数据"""
        end = int(self._tail[1])
        start = max(end - n, 0)
        return self._arr[start:end]
