import contextlib
import os
from typing import Optional, Literal, Tuple

import numpy as np
from loguru import logger
from numpy.lib.format import _write_array_header
from numpy.lib.format import dtype_to_descr

"""
添加的小尾巴，欢迎提供更好的格式方案

1 start 开始位置
2 end 结束位置
3 offset 数据区开始位置，方便其C++语言快速定位并写入
4 magic 魔术数，用来判断是否NPYT格式文件
"""
_TAIL_SIZE_: int = 4
_TAIL_ITEMSIZE_: int = np.dtype(np.uint64).itemsize * _TAIL_SIZE_
_MAGIC_NUMBER_: int = 20250510


class TuplePad(tuple):

    def __repr__(self):
        """npy头信息当想扩充或截断时shape字符串长度固定，提前占位"""
        items = ', '.join(f'{repr(x):>21}' for x in self)
        if len(self) > 1:
            return f"({items})"
        if len(self) == 1:
            return f"({items},)"  # 只有一个元素时，末尾加逗号
        return f"()"  # 空元组


def get_max_shape(shape: tuple, length: int) -> tuple:
    """返回偏大的形状"""
    shape = list(shape)
    if len(shape) == 0:
        shape.append(0)
    shape[0] = max(int(length), shape[0])
    shape = TuplePad(shape)
    return shape


def get_nbytes(dtype: np.dtype, shape: tuple, offset: int) -> int:
    """得到数据区大小"""
    shape = tuple(shape)
    size = np.intp(1)
    for k in shape:
        size *= k

    return int(offset + size * dtype.itemsize)


def get_end(val, end: Optional[int] = None) -> int:
    """数据区结束位置。end为None时，返回val"""
    if end is None:
        end = val
    return int(end)


def get_file_ctx(file, mode: str = "wb"):
    """文件上下文

    wb: 清空只写
    wb+: 清空读写

    """
    if hasattr(file, 'write'):
        file_ctx = contextlib.nullcontext(file)
    else:
        file = os.fspath(file)
        if not file.endswith('.npy'):
            file = file + '.npy'
        file_ctx = open(file, mode)
    return file_ctx


def write_header(fp, array: np.ndarray, shape: tuple) -> int:
    """写入头"""

    def header_data_from_array_1_0(array):
        d = {'shape': shape}

        if array.flags.c_contiguous:
            d['fortran_order'] = False
        elif array.flags.f_contiguous:
            d['fortran_order'] = True
        else:
            d['fortran_order'] = False

        d['descr'] = dtype_to_descr(array.dtype)
        return d

    _write_array_header(fp, header_data_from_array_1_0(array))
    return fp.tell()


def write_footer(fp, dtype: np.dtype, shape: tuple, end: int, offset: int) -> None:
    """定位 文件头+数据区，然后写入尾巴"""
    # 扩充文件大小
    fp.seek(get_nbytes(dtype, shape, offset), 0)
    # 写入尾巴
    fp.write(np.array([0, int(end), offset, _MAGIC_NUMBER_], dtype=np.uint64).tobytes())


def save(file_ctx, array: np.ndarray, length: int, end: Optional[int] = None) -> int:
    # 重新生成新shape
    shape = get_max_shape(array.shape, length)
    end = get_end(array.shape[0], end)

    with file_ctx as fp:
        # 写入头信息
        offset = write_header(fp, array, shape)
        if end > 0:
            # 写入数据
            array.tofile(fp)
        write_footer(fp, array.dtype, shape, end, offset)
        fp.flush()

    return end


def load(filename, mode: Literal["r", "r+", "w+"]) -> Tuple[np.ndarray, np.ndarray]:
    """加载带尾巴的NPY格式文件"""
    arr = np.load(filename, mmap_mode=mode)
    tail = np.memmap(filename, shape=(_TAIL_SIZE_,), dtype=np.uint64, mode=mode,
                     offset=os.path.getsize(filename) - _TAIL_ITEMSIZE_)

    if tail[3] != _MAGIC_NUMBER_:
        logger.error(f"文件格式错误，不是`NPYT`格式文件，涉及到尾部信息的函数都不正确，谨慎使用")
    return arr, tail


def resize(filename: str, row: np.ndarray, end: int, length: Optional[int] = None):
    """文件截断或扩充

    Parameters
    ----------
    filename:str
        文件名
    row:
        初始数据，只是提取dtype等使用。一般只取了一行。
    end:int
        结束位置
    length:Optional[int]
        最大记录长度。None表示使用array的长度。
        当length大于array的长度时，会自动扩充。
        当length小于array的长度时，会自动截断。

    Notes
    -----
    独占时才能使用

    """
    # 自定义数组，不保存数据区，但需要一些基本信息
    shape = get_max_shape(row[:1].shape, length)
    end = get_end(row.shape[0], min(end, shape[0]))

    with get_file_ctx(filename, mode="r+b") as fp:
        offset = write_header(fp, row, shape)
        fp.seek(get_nbytes(row.dtype, shape, 0), 1)
        write_footer(fp, row.dtype, shape, end, offset)
        fp.truncate(fp.tell())
        fp.flush()

    return
