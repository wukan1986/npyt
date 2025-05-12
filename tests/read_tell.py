import os

import numpy as np

from npyt import NPYT

file = "tmp.npy"
arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)


def test_read():
    nt = NPYT(file).save(arr, capacity=6).load(mmap_mode="r+")

    nt._test(5, 3)
    nt.rewind()
    assert nt.tell() == 5
    for i in range(0, 8):
        print(nt.read(n=1))

    print("=" * 60)
    nt.rewind()
    for i in range(0, 4):
        print(nt.read(n=2))

    del nt

    os.remove(file)
