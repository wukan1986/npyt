import os

import numpy as np

from npyt import NPYT

file = "tmp.npy"
arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)


def test_pop():
    nt = NPYT(file).save(arr, capacity=6).load(mmap_mode="r+")

    np.testing.assert_array_equal(nt.head(4), np.array([1, 2, 3, 4], dtype=np.uint64))
    np.testing.assert_array_equal(nt.tail(4), np.array([3, 4, 5, 6], dtype=np.uint64))

    nt._test(4, 2)
    np.testing.assert_array_equal(nt.head(4), np.array([5, 6, 1, 2], dtype=np.uint64))
    np.testing.assert_array_equal(nt.tail(4), np.array([5, 6, 1, 2], dtype=np.uint64))

    nt._test(4, 0)
    np.testing.assert_array_equal(nt.head(4), np.array([5, 6], dtype=np.uint64))
    np.testing.assert_array_equal(nt.tail(4), np.array([5, 6], dtype=np.uint64))

    del nt

    os.remove(file)
