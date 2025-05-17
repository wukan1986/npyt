import os

import numpy as np

from npyt import NPYT

file = "tmp.npy"
arr = np.arange(0, 20, dtype=np.uint64)


def test_read():
    nt = NPYT(file).save(arr).load(mmap_mode="r+")

    nt._test(15, 10)
    nt.rewind()
    assert nt.tell() == 0
    for i in range(0, 8):
        print(nt.read(n=1))

    del nt

    os.remove(file)
