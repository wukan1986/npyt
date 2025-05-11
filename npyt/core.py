# C++ 浅谈Ring Buffer
# https://mp.weixin.qq.com/s/z2JzgS8dt04SJgWPT1v24w
from typing import Optional

import numpy as np
from typing_extensions import Literal  # 3.8+
from typing_extensions import Self  # 3.11+

from npyt.format import get_file_ctx, save, load, resize


class NPYT:
    """
    带尾巴的NPY格式文件,后面带小尾巴，为数据开始和结束位置

    - raw: 原始数据区。与内存映射文件循序一致，为最大长度。可以修改。
    - data: 有效数据区。为有效数据区的长度。环形会拼接成整块。可以修改。

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
        self._t: Optional[np.ndarray] = None
        # 容器，空一行不使用
        self._a: Optional[np.ndarray] = None
        self._capacity: int = 0

    def _test(self, start: int, end: int):
        """测试用。强行设置头尾指针"""
        self._t[0] = start
        self._t[1] = end

    def _raw(self) -> np.ndarray:
        """测试用。取原始数组"""
        return self._a

    def _raw_len(self) -> int:
        """测试用。获取原始数组长度"""
        return self._a.shape[0]

    def info(self):
        """获取尾巴关键信息"""
        return tuple(self._t.tolist())

    def clear(self):
        """重置位置指针，相当于清空了数据"""
        self._t[0:2] = 0

    def start(self) -> int:
        """获取缓冲区开始位置"""
        return int(self._t[0])

    def end(self) -> int:
        """获取缓冲区结束位置"""
        return int(self._t[1])

    def empty(self) -> bool:
        """查询缓冲区是否为空"""
        return self.start() == self.end()

    def full(self) -> bool:
        """查询缓冲区是否已满"""
        return (self.end() + 1) % self._capacity == self.start()

    def size(self) -> int:
        """当前缓冲区中元素个数"""
        start = self.start()
        end = self.end()
        if end >= start:
            return end - start
        else:
            return self._capacity - (start - end)

    def capacity(self) -> int:
        """缓冲区容量大小（最大可容纳元素数）"""
        return self._capacity - 1

    def head(self, n: int = 5) -> np.ndarray:
        """取头部数据"""
        return self.data()[:n]

    def tail(self, n: int = 5) -> np.ndarray:
        """取尾部数据"""
        return self.data()[-n:]

    def load(self, mmap_mode: Literal["r", "r+"]) -> Self:
        """加载文件。为以后操作做准备

        Parameters
        ----------
        mmap_mode
            内存文件映射模式

            r: 只读
            r+: 读写

        """
        self._a, self._t = load(self._filename, mmap_mode=mmap_mode)
        # 记录容器容量，空一行不存
        self._capacity = self._a.shape[0]
        return self

    def save(self, array: np.ndarray, capacity: int = 0, end: Optional[int] = None) -> Self:
        """创建文件

        Parameters
        ----------
        array:
            初始数据
        capacity:int
            最大容量
        end:int
            结束位置。0表示只创建文件，数据区为空。None表示使用array的长度。

        """
        save(get_file_ctx(self._filename, mode="wb+"), array, capacity, end)

        return self

    def resize(self, capacity: Optional[int] = None) -> Self:
        """文件截断或扩充。不能丢失有效数据

        Parameters
        ----------
        capacity:int
            新的长度。

            None将截断数据

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
        if capacity is None:
            capacity = end
        capacity = max(end, capacity)

        # 一定要copy,因为后面要释放文件
        arr = self._a[:1].copy()
        # 释放文件占用
        self._a = None
        self._t = None
        # 释放后就可以动文件了
        resize(self._filename, arr, start, end, capacity)

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
            return self._a[start:end]
        elif part == 0:
            # 右端，有一行不用
            return self._a[start:-1]
        elif part == 1:
            # 左端
            return self._a[:end]
        else:
            # 环向。原数据无法修改
            return np.concatenate([self._a[start:-1], self._a[:end]])

    def data(self) -> np.ndarray:
        """取数据区。环形数据会拼接起来不可修改"""
        return self.slice(self.start(), self.end(), part=2)

    def append(self, array: np.ndarray, ringbuffer: bool = False) -> int:
        """普通缓冲区插入函数，满了后可能只插入了部分

        Parameters
        ----------
        array:
            插入的数据
        ringbuffer:bool
            是否RingBuffer模式

        Returns
        -------
        int
            剩余未插入的行数

        Raises
        ------
        ValueError
            插入的数据长度超过了缓冲区长度

        Notes
        -----
        1. 插入一行数据时要注意，shape为(1, n)，而不是(n,)
            - arr[0:1] 正确
            - arr[0] 错误

        2. a[0:0] = b[0:1] 不报错，也没保存
            (0, 3)  (1, 3)

        3. a[10:20] = b[0:1] 不报错，也没保存
            (0, 3)  (1, 3)

        """
        remaining = array.shape[0]
        # 空内容，没必要
        if remaining == 0:
            return remaining

        start = self.start()
        end = self.end()
        if end == self.capacity() and start > 0:
            # 到末尾了,头又有空间，移动到开头
            if ringbuffer:
                end = 0

        # 找到可填充大小
        if end >= start:
            _size = min(self.capacity() - end, remaining)
        else:
            _size = min(start - 1 - end, remaining)

        # 无可填充空间，跳过
        if _size <= 0:
            return remaining
        remaining -= _size

        _end = end + _size
        self._a[end:_end] = array[:_size]
        self._t[1] = _end

        return remaining


class NPYT_RB(NPYT):
    """RingBuffer版"""

    def append(self, array: np.ndarray, ringbuffer: bool = True) -> int:
        """添加数据，默认是环形缓冲区"""
        return super().append(array, ringbuffer)

    def append2(self, array: np.ndarray) -> int:
        """执行两次，第一次填充右边，第二次填充左边"""
        remaining = self.append(array, ringbuffer=True)
        if remaining > 0:
            remaining = self.append(array[-remaining:], ringbuffer=True)
        return remaining

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
            arr = self._a[start:end]
            if copy:
                arr = arr.copy()
            self._t[0] = end  # 在任意位置
        else:
            # 要取两次，第一次到
            arr = self._a[start:self.capacity()]
            if copy:
                arr = arr.copy()
            self._t[0] = 0

        return arr
