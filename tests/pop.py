import os

import numpy as np

from npyt import NPYT_RB

file = "tmp.npy"
arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)


def test_pop():
    nt = NPYT_RB(file).save(arr, capacity=6).load(mmap_mode="r+")

    np.testing.assert_array_equal(nt.pop(copy=False), arr)
    assert len(nt.pop(copy=False)) == 0

    nt._test(4, 2)
    np.testing.assert_array_equal(nt.data(), np.array([5, 6, 1, 2], dtype=np.uint64))
    np.testing.assert_array_equal(nt.pop(copy=False), np.array([5, 6], dtype=np.uint64))
    np.testing.assert_array_equal(nt.pop(copy=False), np.array([1, 2], dtype=np.uint64))

    del nt

    os.remove(file)
