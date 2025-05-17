import os

import numpy as np

from npyt import NPYT

file1 = "tmp1.npy"
file2 = "tmp2.npy"
arr = np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)

nt1 = NPYT(file1).save(arr, capacity=10).load(mmap_mode="r+")
nt2 = NPYT(file2).save(arr, capacity=10).load(mmap_mode="r+")

assert nt1.append(arr) == 6
print(nt1.info())
assert nt1.append(arr[-4:]) == 0
nt1.merge(nt2)

nt1.rename("A.npy")

os.remove("A.npy")


