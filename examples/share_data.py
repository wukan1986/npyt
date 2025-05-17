import os

import numpy as np

from npyt import NPYT

arr = np.array([1, 2, 3, 4, 5, 6])

file = "tmp.npy"
# 创建文件
nt1 = NPYT(file).save(arr, capacity=10, end=0).load(mmap_mode="r+")
# 只读加载文件
nt2 = NPYT(file).load(mmap_mode="r")
nt3 = NPYT(file).load(mmap_mode="r")

nt1.append(arr)
print(nt2.data())

nt1.append(arr[0:1])
print(nt3.data())

nt1.data()[:3] = 0
print(nt3.data())

del nt1
del nt2
del nt3
os.remove(file)
