import shutil
import time
from pathlib import Path
from typing import List, Optional

import more_itertools
import numpy as np
from loguru import logger
from typing_extensions import Self

from npyt import NPYT


class NPY8:

    def __init__(self, name: str, capacity_per_file: int = 1024, query_size: int = 4):
        """无尽头增长文件

        Parameters
        ----------
        name:str
            目录名
        capacity_per_file:int
            每个子文件最大容量
        query_size:int
            队列长度

        """
        self._name: str = name
        self._capacity_per_file: int = capacity_per_file
        self._size: int = max(query_size, 2)

        self._path: Path = Path(self._name)
        self._path.mkdir(parents=True, exist_ok=True)
        self._path_lock = self._path / '.lock'
        # lock文件，维护了多个时间戳文件名
        self._lock: Optional[np.ndarray] = None
        self._writer: Optional[NPYT] = None
        self._reader: Optional[NPYT] = None
        # 正在读的文件名时间戳
        self._reader_ts: int = -1

    def remove(self):
        """删除文件"""
        self._writer = None
        self._reader = None
        self._lock = None
        shutil.rmtree(self._path)
        logger.info("remove {} ", self._path.resolve())

    def load(self) -> Self:
        """初始化并加载"""
        if self._path_lock.exists():
            self._lock = np.memmap(self._path_lock, dtype=np.uint64, mode="r+", shape=(self._size,))
        else:
            self._lock = np.memmap(self._path_lock, dtype=np.uint64, mode="w+", shape=(self._size,))

        # 没有数据，初始化
        t = self._lock[0]
        filename = self._path / f'{t}.npy'
        if not filename.exists():
            files = sorted(self._path.glob('*.npy'))[-self._size:]
            self._lock[:] = 0
            for i, f in enumerate(files):
                self._lock[i] = int(f.stem)

        return self

    def append(self, data: np.ndarray) -> Self:
        """添加数据，遇到文件空间不足时会新增文件

        Parameters
        ----------
        data:np.ndarray
            新数据

        """
        if self._writer:
            remaining = self._writer.append(data)
            if remaining == 0:
                # 成功
                return self

            # 失败
            if self._lock[-1] > 0:
                # 队列满了，平移队列
                t = self._lock[0]
                self._lock[:-1] = self._lock[1:]
                self._lock[-1] = time.time_ns()
                # 出队列后修改文件大小。留心文件被占用导致失败
                filename = self._path / f'{t}.npy'
                if t == self._reader_ts:
                    self._reader = None
                    logger.warning('{} is opened', filename.resolve())
                NPYT(filename).load(mmap_mode='r').resize()
            else:
                # 到下一个位置
                self._lock[np.argmax(self._lock) + 1] = time.time_ns()

        # 这样基本不会有空文件
        if self._lock[0] == 0:
            self._lock[0] = time.time_ns()

        # 找到最大编号文件。但不知道文件是否满了
        t = self._lock[np.argmax(self._lock)]
        filename = self._path / f'{t}.npy'
        if filename.exists():
            # 加载已有文件
            self._writer = NPYT(filename).load(mmap_mode="r+")
            return self.append(data)
        else:
            logger.trace("create {}", filename.resolve())
            # 可以一次性保存大文件
            self._writer = NPYT(filename).save(array=data, capacity=self._capacity_per_file).load(mmap_mode="r+")
            return self

    def read(self, n: int = 1024, prefetch: int = 0) -> np.ndarray:
        """读取数据

        Parameters
        ----------
        n:int
            从当前位置，读取行数
        prefetch:int
            预加载行数

        Notes
        -----
        一次性取数限制在单个文件，当前文件读完后，再次读取自动切换成下一文件

        """
        if self._reader:
            arr = self._reader.read(n, prefetch)
            if len(arr) > 0:
                # 成功
                return arr

            # 最后一个文件，直接返回，下次要用
            if np.max(self._lock) == self._reader_ts:
                return np.empty(0)

        while True:
            max_idx = np.argmax(self._lock > self._reader_ts)
            if max_idx == 0:
                if np.max(self._lock) == self._reader_ts:
                    return np.empty(0)
            # 找到最大编号文件。但不知道文件是否满了
            t = self._lock[max_idx]
            filename = self._path / f'{t}.npy'
            self._reader_ts = t
            if filename.exists():
                # 加载已有文件
                self._reader = NPYT(filename).load(mmap_mode="r")
                return self.read(n, prefetch)

    def tail(self, n: int = 5) -> List[np.ndarray]:
        """取尾部数据

        Parameters
        ----------
        n:int
            行数

        Returns
        -------
        List[np.ndarray]
            一个文件对应一个np.ndarray

        """
        max_idx = np.argmax(self._lock)
        outputs = []
        remaining = n
        for i in range(max_idx, -1, -1):
            t = self._lock[i]
            filename = self._path / f'{t}.npy'
            if filename.exists():
                arr = NPYT(filename).load(mmap_mode="r").tail(remaining)
                outputs.insert(0, arr)
                remaining -= len(arr)
                if remaining <= 0:
                    break

        return outputs

    def head(self, n: int = 5) -> List[np.ndarray]:
        """取头部数据

        Parameters
        ----------
        n:int
            行数

        Returns
        -------
        List[np.ndarray]
            一个文件对应一个np.ndarray

        """
        max_idx = np.argmax(self._lock)
        outputs = []
        remaining = n
        for i in range(0, max_idx):
            t = self._lock[i]
            filename = self._path / f'{t}.npy'
            if filename.exists():
                arr = NPYT(filename).load(mmap_mode="r").tail(remaining)
                outputs.append(arr)
                remaining -= len(arr)
                if remaining <= 0:
                    break

        return outputs

    def merge(self, batch_size: int = 4):
        """合并文件，并改名

        Parameters
        ----------
        batch_size:int
            每批次大小

        """
        batch_size = max(batch_size, 2)
        files = sorted(self._path.glob('*.npy'))[:-self._size]
        for batch in more_itertools.batched(files, batch_size):
            if len(batch) != batch_size:
                continue
            f1 = NPYT(batch[0]).load(mmap_mode="r+")
            for f in batch[1:]:
                f1.merge(NPYT(f).load(mmap_mode="r"))
            # 改个名字，防止重复合并
            f = f1.filename().with_suffix('.npy_')
            f1.rename(f)
            logger.info("merge to {} from {}", f, batch)
