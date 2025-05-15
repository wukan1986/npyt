# C++ 浅谈Ring Buffer
# https://mp.weixin.qq.com/s/z2JzgS8dt04SJgWPT1v24w
import os
import pathlib
import shutil
from datetime import datetime
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
        self._tell: int = 0
        # 从np.save/np.load丢弃了重要的alignment
        self._dtype: Optional[np.dtype] = None

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
        if self._t is None:
            return None
        return tuple(self._t.tolist())

    def dtype(self) -> np.dtype:
        """获取数据类型"""
        return self._dtype

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
        return (self.end() + 1) % (self._capacity + 1) == self.start()

    def size(self) -> int:
        """当前缓冲区中元素个数"""
        start, end = self.start(), self.end()
        if end >= start:
            return end - start
        else:
            return self._capacity - (start - end)

    def capacity(self) -> int:
        """缓冲区容量大小（最大可容纳元素数）"""
        return self._capacity

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
        self._capacity = self._a.shape[0]
        if self._dtype is None:
            self._dtype = self._a.dtype
        return self

    def save(self, array: Optional[np.ndarray] = None, dtype: Optional[np.dtype] = None,
             capacity: int = 0, end: Optional[int] = None,
             skip_if_exists: bool = True) -> Self:
        """创建文件

        Parameters
        ----------
        array: np.ndarry
            初始数据.如果为None，将创建一个空数组。
        dtype:np.dtype
            数据类型
        capacity:int
            最大容量
        end:int
            结束位置。0表示只创建文件，数据区为空。None表示使用array的长度。
        skip_if_exists:bool
            如果文件已经存在了就跳过。反之新建

        """
        if skip_if_exists and os.path.exists(self._filename):
            return self

        if array is None:
            array = np.empty((1,), dtype=dtype)
            end = 0
        # 记下来，等会可能用到
        self._dtype = array.dtype
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
        start, end = self.start(), self.end()

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

    def backup(self, to_path: str, dt: datetime = datetime.now()) -> None:
        """备份"""
        path = pathlib.Path(to_path) / dt.strftime("%Y%m%d")
        path.mkdir(parents=True, exist_ok=True)

        shutil.copy2(self._filename, path)

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
            # 右端
            return self._a[start:]
        elif part == 1:
            # 左端
            return self._a[:end]
        else:
            # 环向。原数据无法修改
            return np.concatenate([self._a[start:], self._a[:end]])

    def data(self) -> np.ndarray:
        """取数据区。环形数据会拼接起来不可修改"""
        return self.slice(self.start(), self.end(), part=2)

    def head(self, n: int = 5) -> np.ndarray:
        """取头部数据"""
        # 这种写法某些情况下产生复制，效率低
        # return self.data()[:n]
        start, end = self.start(), self.end()
        if end >= start:
            return self._a[start:min(start + n, end)]

        a1 = self._a[start:start + n]
        remaining = n - len(a1)
        if remaining == 0 or end == 0:
            return a1

        # 这后面发生了合并复制
        a2 = self._a[:min(remaining, end)]
        return np.concatenate([a1, a2])

    def tail(self, n: int = 5) -> np.ndarray:
        """取尾部数据"""
        # 这种写法某些情况下产生复制，效率低
        # return self.data()[-n:]
        start, end = self.start(), self.end()
        if end >= start:
            return self._a[max(start, end - n):end]

        a2 = self._a[max(end - n, 0):end]
        remaining = n - len(a2)
        if remaining == 0 or start == self._raw_len():
            return a2

        # 这后面发生了合并复制
        a1 = self._a[max(self._raw_len() - remaining, start):]
        return np.concatenate([a1, a2])

    def at(self, index):
        return self._a[index]

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
        start, end = self.start(), self.end()
        if end >= start:
            _end = end
            _start = end
        else:
            _end = self._raw_len()
            _start = 0

        arr = self._a[start:_end]
        if copy:
            arr = arr.copy()
        self._t[0] = _start

        return arr

    def append(self, array: np.ndarray, ringbuffer: bool = False, bulk: bool = True) -> int:
        """缓冲区插入函数

        Parameters
        ----------
        array:
            插入的数据
        ringbuffer:bool
            是否RingBuffer模式。RingBuffer模式会出现start>end的情况，在取全体数据时会导致拼接复制
        bulk:bool
            整体一批插入，不能分两次。

            - True: 空间不够时，不插入数据，返回原数组长度
            - False: 空间不够时，能插就插，返回未插入长度

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
        1. 插入单行数据时要注意，shape为(1, n)，而不是(n,)
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

        start, end = self.start(), self.end()

        # 数据空了，后面空间也不够，移动到开头
        if start == end and end + remaining >= self._raw_len():
            end = 0
            start = 0
        elif end == self._raw_len():
            # 到末尾了,头部又有空间，移动到开头
            if ringbuffer and start > 0:
                end = 0

        # 找到可填充大小
        if end >= start:
            _size = min(self._raw_len() - end, remaining)
        else:
            _size = min(start - 1 - end, remaining)

        if bulk and _size < remaining:
            # 数据必须整体插入，不能分成两部分
            return remaining

        # 无可填充空间，跳过
        if _size <= 0:
            return remaining
        remaining -= _size

        _end = end + _size
        self._a[end:_end] = array[:_size]
        self._t[1] = _end

        return remaining

    def tell(self) -> int:
        return self._tell

    def rewind(self) -> None:
        """重置当前指针到数据的起始位置

        Notes
        -----
        `pop`后导致start发生变化，根据业务需要，可能要重置指针到start

        """
        self._tell = self.start()

    def seek(self, offset: int, whence: int = 0) -> None:
        """在start:end范围内seek

        Parameters
        ----------
        offset:int
            >0: 向后偏
            <0: 向前偏
        whence:int
            0,1,2

        """
        start, end = self.start(), self.end()
        if whence == os.SEEK_SET:
            _curr = start
        elif whence == os.SEEK_CUR:
            _curr = self._tell
        elif whence == os.SEEK_END:
            _curr = end
        else:
            _curr = end

        if end >= start:
            self._tell = max(min(_curr + offset, end), start)
        else:
            start -= self._raw_len()
            end += self._raw_len()
            self._tell = max(min(_curr + offset, end), start) % (self._capacity + 1)

    def read(self, n: int = 1, prefetch: int = 0, copy: bool = False) -> np.ndarray:
        """读取n行数据。不移动start指针，而是移动tell指针

        Parameters
        ----------
        n:int
            读取行数
        prefetch:int
            预读取行数。需>=0
        copy:bool
            是否复制

        Returns
        -------
        np.ndarray
            读取的数据

        """
        start, end = self.start(), self.end()
        if end >= start:
            _start = max(self._tell - prefetch, start)
            _end = min(self._tell + n, end)
            _tell = _end
        elif end >= self._tell:
            # 虽有预取，但不折返取
            _start = max(self._tell - prefetch, 0)
            _end = min(self._tell + n, end)
            _tell = _end
        else:
            _start = max(self._tell - prefetch, start)
            _end = min(self._tell + n, self._raw_len())
            # 指针移动到开头
            _tell = _end % self._capacity

        arr = self._a[_start:_end]
        if copy:
            arr = arr.copy()
        self._tell = _tell

        return arr


class NPYT_RB(NPYT):
    """RingBuffer版"""

    def append(self, array: np.ndarray, ringbuffer: bool = True, bulk: bool = False) -> int:
        """添加数据，默认是环形缓冲区"""
        return super().append(array, ringbuffer, bulk)

    def append2(self, array: np.ndarray) -> int:
        """执行两次，第一次填充右边，第二次填充左边"""
        remaining = self.append(array, ringbuffer=True, bulk=False)
        # 插入失败，直接返回
        if remaining == array.shape[0]:
            return remaining

        # 还有剩余，再插一次
        if remaining > 0:
            remaining = self.append(array[-remaining:], ringbuffer=True, bulk=False)
        return remaining

    def pop2(self, copy=True):
        arr1 = super().pop(copy=copy)
        arr2 = super().pop(copy=copy)
        if arr2.shape[0] == 0:
            return arr1
        return np.concatenate([arr1, arr2])
