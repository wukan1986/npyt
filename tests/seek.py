import os

import numpy as np

from npyt import NPYT

file = "tmp.npy"
arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)


def test_pop():
    nt = NPYT(file).save(arr, capacity=6).load(mmap_mode="r+")

    nt.seek(6, 0)
    assert nt.tell() == 6
    nt.seek(7, 0)
    assert nt.tell() == 6
    nt.seek(-6, 2)
    assert nt.tell() == 0
    nt.seek(-7, 2)
    assert nt.tell() == 0

    nt._test(5, 3)
    for i in range(1, 6):
        nt.seek(i, 0)
        print(nt.tell())

    print("=" * 60)

    for i in range(-1, -6, -1):
        nt.seek(i, 2)
        print(nt.tell())

    del nt

    os.remove(file)
