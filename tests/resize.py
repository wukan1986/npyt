import os

import numpy as np

from npyt import NPYT

file = "tmp.npy"
arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)


def test_resize():
    nt = NPYT(file).save(arr, capacity=10).load(mmap_mode="r+")
    nt.resize(capacity=None).load(mmap_mode="r")
    np.testing.assert_array_equal(nt._raw(), arr)

    nt.resize(capacity=2).load(mmap_mode="r")
    np.testing.assert_array_equal(nt._raw(), arr)

    nt.resize(capacity=8).load(mmap_mode="r")
    # 多了原来info中的信息，正好是0,6
    arr1 = np.array([1, 2, 3, 4, 5, 6, 0, 6], dtype=np.uint64)
    np.testing.assert_array_equal(nt._raw(), arr1)

    nt2 = NPYT(file).load(mmap_mode="r+")
    nt2.data()
    nt.resize(capacity=20).load(mmap_mode="r")
    print(nt2._raw())
    nt.resize(capacity=None).load(mmap_mode="r")
    print(nt2._raw())

    del nt
    del nt2

    os.remove(file)
