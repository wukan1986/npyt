import os

import numpy as np

from npyt import NPYT

file = "tmp.npy"
arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)


def test_append():
    nt = NPYT(file).save(arr, capacity=10).load(mmap_mode="r+")

    assert nt.append(arr, ringbuffer=False, bulk=True) == 6
    assert nt.append(arr, ringbuffer=False, bulk=False) == 2
    print(nt.info())
    assert nt.append(arr[-2:], ringbuffer=True, bulk=False) == 2

    nt._test(7, 10)
    assert nt.append(arr[-2:], ringbuffer=True, bulk=False) == 0
    print(nt._raw())
    assert nt.append(arr, ringbuffer=True, bulk=False) == 2
    print(nt._raw())

    del nt

    os.remove(file)
