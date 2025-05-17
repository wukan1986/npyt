# C++ 浅谈Ring Buffer
# https://mp.weixin.qq.com/s/z2JzgS8dt04SJgWPT1v24w
import os
import shutil
from pathlib import Path
from typing import Optional, Union

import numpy as np
from loguru import logger
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

    def __init__(self, filename: Union[str, Path], dtype: Optional[np.dtype] = None):
        """初始化

        Parameters
        ----------
        filename:str
            文件名
        dtype:np.dtype
            数据类型。
                to_records时，数据类型可能与预期的不一致。append时可能取的dtyep不满足。所以在这提前限制更合适

        """
        self._filename: Path = Path(filename)
        self._t: Optional[np.ndarray] = None
        # 容器，空一行不使用
        self._a: Optional[np.ndarray] = None
        self._capacity: int = 0
        self._tell: int = 0
        # 从np.save/np.load丢弃了重要的alignment
        self._dtype: Optional[np.dtype] = dtype

    def filename(self) -> Path:
        return self._filename

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

    def clear(self) -> None:
        """重置位置指针，相当于清空了数据"""
        self._t[0:2] = 0

    def start(self) -> int:
        """获取缓冲区开始位置"""
        # return int(self._t[0])
        return 0

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
        else:
            assert self._dtype == self._a.dtype, f"dtype mismatch {self._dtype} != {self._a.dtype}"
        return self

    def save(self, array: Optional[np.ndarray] = None,
             capacity: int = 0, end: Optional[int] = None,
             skip_if_exists: bool = True) -> Self:
        """创建文件

        Parameters
        ----------
        array: np.ndarry
            初始数据.如果为None，将创建一个空数组。
        capacity:int
            最大容量
        end:int
            结束位置。0表示只创建文件，数据区为空。None表示使用array的长度。
        skip_if_exists:bool
            如果文件已经存在了就跳过。反之新建

        """
        if skip_if_exists and self._filename.exists():
            return self

        if array is None:
            array = np.empty((1,), dtype=self._dtype)
            end = 0
        elif self._dtype is None:
            self._dtype = array.dtype
        else:
            assert self._dtype == array.dtype, f"dtype mismatch {self._dtype} != {array.dtype}"
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

    def backup(self, to_path: str) -> None:
        """备份"""
        path = Path(to_path)
        path.mkdir(parents=True, exist_ok=True)

        shutil.copy2(self._filename, path)
        logger.info("backup {} to {}", self._filename, path)

    def data(self) -> np.ndarray:
        """取数据区。环形数据会拼接起来不可修改"""
        start, end = self.start(), self.end()
        return self._a[start:end]

    def head(self, n: int = 5) -> np.ndarray:
        """取头部数据"""
        start, end = self.start(), self.end()
        return self._a[start:min(start + n, end)]

    def tail(self, n: int = 5) -> np.ndarray:
        """取尾部数据"""
        start, end = self.start(), self.end()
        return self._a[max(start, end - n):end]

    def at(self, index):
        return self._a[index]

    def append(self, array: np.ndarray) -> int:
        """缓冲区插入函数

        Parameters
        ----------
        array:
            插入的数据

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

        end = self.end()
        _end = end + remaining
        # 数据空了，后面空间也不够，移动到开头
        if _end > self._raw_len():
            return remaining

        self._a[end:_end] = array
        self._t[1] = _end

        return 0

    def expend(self, array: np.ndarray) -> bool:
        """缓冲区插入函数，空间不够扩充文件大小

        Parameters
        ----------
        array:
            插入的数据

        Returns
        -------
        int
            剩余未插入的行数

        """
        remaining = array.shape[0]
        # 空内容，没必要
        if remaining == 0:
            return False

        end = self.end()
        _end = end + remaining
        # 数据空了，后面空间也不够，移动到开头
        if _end > self._raw_len():
            if self.resize(_end):
                self.load("r+").append(array)
                return True
            else:
                self.load("r+")
                return False

        self._a[end:_end] = array
        self._t[1] = _end

        return True

    def remove(self) -> bool:
        """删除文件"""
        self._a = None
        self._t = None
        try:
            os.remove(self._filename)
            logger.trace("remove {}", self._filename.resolve())
            return True
        except PermissionError as e:
            logger.error("remove {} error:{}", self._filename.resolve(), e)
            return False

    def merge(self, obj: Self) -> bool:
        """合并文件"""
        if self.expend(obj.data()):
            return obj.remove()
        return False

    def rename(self, name) -> Self:
        """重命名"""
        self._a = None
        self._t = None
        shutil.move(self._filename, name)
        self._filename = Path(name)
        return self

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

        self._tell = max(min(_curr + offset, end), start)

    def read(self, n: int = 1, prefetch: int = 0) -> np.ndarray:
        """读取n行数据。不移动start指针，而是移动tell指针

        Parameters
        ----------
        n:int
            读取行数
        prefetch:int
            预读取行数。需>=0

        Returns
        -------
        np.ndarray
            读取的数据

        """
        start, end = self.start(), self.end()

        _start = max(self._tell - prefetch, start)
        self._tell = min(self._tell + n, end)

        arr = self._a[_start:self._tell]

        return arr
