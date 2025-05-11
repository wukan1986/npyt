from typing import Literal, Optional

import numpy as np

from npyt.format import get_file_ctx, save, load, resize

__all__ = [
    "NPYT",
]


class NPYT:
    """
    带尾巴的NPY格式文件,后面带小尾巴，为数据开始和结束位置

    - raw: 原始数据区。与内存映射文件循序一致，为最大长度。可以修改。
    - data: 有效数据区。为有效数据区的长度。环形会拼接成整块。可以修改。
    - buffer: 缓冲区。从启

    Notes
    -----
    很多操作没有条件判断，使用时需要注意。

    """

    def __init__(self, filename: str):
        """初始化

        Parameters
        ----------
        filename:str
            文件名

        """
        self._filename: str = filename
        self._arr: Optional[np.ndarray] = None
        self._tail: Optional[np.ndarray] = None
        self._len: int = 0

    def _test(self, start: int, end: int):
        """强行设置头尾指针，仅用于测试"""
        self._tail[0] = start
        self._tail[1] = end

    def info(self):
        """获取尾巴关键信息"""
        return tuple(self._tail.tolist())

    def clear(self):
        """重置位置指针，相当于清空了数据"""
        self._tail[0:2] = 0

    def start(self) -> int:
        """获取数据区开始位置"""
        return int(self._tail[0])

    def end(self) -> int:
        """获取数据区结束位置"""
        return int(self._tail[1])

    def is_empty(self) -> bool:
        """判断是否为空"""
        return self.start() == self.end()

    def is_full(self) -> bool:
        """判断是否已满"""
        start = self.start()
        end = self.end()
        if start == 0:
            # 正向。可以存满
            return end == self._len
        else:
            # 环向。留一个位置不存，否则无法区分是否为空
            return end + 1 == start

    def head(self, n: int = 5) -> np.ndarray:
        """取头部数据"""
        return self.data()[:n]

    def tail(self, n: int = 5) -> np.ndarray:
        """取尾部数据"""
        return self.data()[-n:]

    def load(self, mmap_mode: Literal["r", "r+"]) -> "NPYT":
        """加载文件。为以后操作做准备

        Parameters
        ----------
        mmap_mode
            内存文件映射模式

            r: 只读
            r+: 读写

        """
        self._arr, self._tail = load(self._filename, mmap_mode=mmap_mode)
        # 记录两个重要的长度
        self._len = self._arr.shape[0]
        return self

    def save(self, array: np.ndarray, length: int = 0, end: Optional[int] = None) -> "NPYT":
        """创建文件

        Parameters
        ----------
        array:
            初始数据
        length:int
            记录长度
        end:int
            结束位置。0表示只创建文件，数据区为空。None表示使用array的长度。

        """
        save(get_file_ctx(self._filename, mode="wb+"), array, length, end)

        return self

    def resize(self, length: Optional[int] = None) -> "NPYT":
        """文件截断或扩充。不能丢失有效数据

        Parameters
        ----------
        length:int
            新的长度。

            None将默认截取到有效数据右边界

        Notes
        -----
        独占时才能使用

        """
        # 一些基本信息
        start = self.start()
        end = self.end()

        # 数据环形，文件不能动了
        if end < start:
            return self

        # 有效右边界
        if length is None:
            length = end
        length = max(end, length)

        # 一定要copy,因为后面要释放文件
        arr = self._arr[:1].copy()
        # 释放文件占用
        self._arr = None
        self._tail = None
        # 释放后就可以动文件了
        resize(self._filename, arr, start, end, length)

        return self

    def slice(self, start: int, end: int, part: int) -> np.ndarray:
        """切片。数据出现拼接时将无法直接修改

        Parameters
        ----------
        start:int
        end:int
        part:int

        Notes
        -----
        主要是内部使用，所以没有参数检查

        """
        if end >= start:
            # 正向。原数据可被修改
            return self._arr[start:end]
        elif part == 0:
            # 右端
            return self._arr[start:]
        elif part == 1:
            # 左端
            return self._arr[:end]
        else:
            # 环向。原数据无法修改
            return np.concatenate([self._arr[start:], self._arr[:end]])

    def append(self, array: np.ndarray) -> "NPYT":
        """插入数据

        Parameters
        ----------
        array:
            插入的数据

        Raises
        ------
        ValueError
            插入的数据长度超过了缓冲区长度

        Notes
        -----
        1. 插入一行数据时要注意，shape为(1, n)，而不是(n,)
            - arr[0:1] 正确
            - arr[0] 错误

        2. a[0:0] = b[0:1] 不报错
            (0, 3)  (1, 3)

        3. a[10:20] = b[0:1] 不报错
            (0, 3)  (1, 3)

        """
        size = array.shape[0]
        if size == 0:
            # 插入空，直接返回
            return self

        _start = self.end()
        _end = _start + size

        self._arr[_start:_end] = array  # ValueError
        self._tail[1] = _end

        return self

    def raw(self) -> np.ndarray:
        """取原始数组"""
        return self._arr

    def raw_len(self) -> int:
        """获取原始数组长度"""
        return self._len

    def data(self) -> np.ndarray:
        """取数据区。环形数据会拼接起来不可修改"""
        return self.slice(self.start(), self.end(), part=2)

    def data_len(self) -> int:
        """获取数组长度"""
        length = self.end() - self.start()
        if length >= 0:
            # 正向
            return length
        else:
            # 环向
            return self._len + length

    def pop(self, copy: bool) -> np.ndarray:
        """普通缓冲区不应当有pop操作"""
        raise NotImplementedError


class NPYT_RB(NPYT):
    """RingBuffer版"""

    def right_idx(self) -> int:
        """取最右位置

        Notes
        -----
        start左侧部分不考虑

        """
        start = self.start()
        end = self.end()
        if end >= start:
            # 正向
            return self._len
        else:
            # 环向
            return start - 1

    def pop(self, copy: bool) -> np.ndarray:
        """取数据块。环形要取两次才能取完

        底层是切片，然后移动指针位置

        Parameters
        ----------
        copy:bool
            是否复制

        Notes
        -----
        1. 取数时，如果其他线程正在写，可能指针位置已经指向开头，但数据还没有复制出来
         可以选择`copy=True`，但遇到大数据量性能下降

        """
        start = self.start()
        end = self.end()
        if end >= start:
            arr = self._arr[start:end]
            if copy:
                arr = arr.copy()
            if end >= self._len:
                # 末尾，移动到开头
                self._tail[0:2] = 0
            else:
                # 中段，移动到结束位置
                self._tail[0] = end
        else:
            # 分两次
            arr = self._arr[start:]
            if copy:
                arr = arr.copy()
            self._tail[0] = 0

        return arr

    def buffer(self, ringbuffer: bool) -> np.ndarray:
        """获取缓冲区"""
        if ringbuffer:
            # 环向。留一个位置
            start = self.start()
            return self.slice(start, start - 1, part=2)
        else:
            # 正向。全长
            return self._arr

    def buffer_len(self, ringbuffer: bool) -> int:
        """获取缓冲区长度"""
        if ringbuffer:
            # 环向。留一个位置
            return self._len - 1
        else:
            # 正向。全长
            return self._len

    def free_len(self, ringbuffer) -> int:
        """剩余空间"""
        if ringbuffer:
            # 环形缓冲区
            return self.buffer_len(ringbuffer) - self.data_len()
        else:
            # 正向缓冲区
            return self.buffer_len(ringbuffer) - self.end()

    def append(self, array: np.ndarray, ringbuffer: bool = False) -> "NPYT":
        """插入数据

        Parameters
        ----------
        array:
            插入的数据
        ringbuffer:bool
            是否环形缓冲区模式

        Raises
        ------
        ValueError
            插入的数据长度超过了缓冲区长度

        Notes
        -----
        1. 插入一行数据时要注意，shape为(1, n)，而不是(n,)
            - arr[0:1] 正确
            - arr[0] 错误

        2. a[0:0] = b[0:1] 不报错
            (0, 3)  (1, 3)

        """
        size = array.shape[0]
        if size == 0:
            # 插入空，直接返回
            return self

        _start = self.end()
        _end = _start + size
        if ringbuffer:
            start = self.start()
            end = self.end()
            if end >= start:
                size2 = _end - self.right_idx()
                size1 = size - size2
                self._arr[0:max(min(size2, start - 1), 0)] = array[size1:]  # 不报错呀
                self._arr[_start:] = array[0:size1]
            else:
                pass
        else:
            self._arr[_start:_end] = array  # ValueError
            self._tail[1] = _end

        return self
