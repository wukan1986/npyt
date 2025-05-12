import os

import numpy as np

from npyt import NPYT_RB

file = "tmp.npy"
arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)


def test_ringbuffer():
    nt = NPYT_RB(file).save(arr, capacity=6).load(mmap_mode="r+")

    nt._test(4, 2)
    # pop了两次，指针到了2,2
    np.testing.assert_array_equal(nt.pop2(), np.array([5, 6, 1, 2], dtype=np.uint64))
    # 放不下，空白重置0,0
    nt.append2(arr)
    np.testing.assert_array_equal(nt._raw(), arr)

    del nt

    os.remove(file)
