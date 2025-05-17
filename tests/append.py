import os

import numpy as np

from npyt import NPYT

file = "tmp.npy"
arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)


def test_append():
    nt = NPYT(file).save(arr, capacity=10).load(mmap_mode="r+")

    assert nt.append(arr) == 6
    print(nt.info())
    assert nt.append(arr[-4:]) == 0
    assert nt.append(arr[-1:]) == 1

    del nt

    os.remove(file)
