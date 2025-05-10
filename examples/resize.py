import numpy as np

from npyt import NPYT

arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64) + 64

# 创建文件
nt = NPYT("f3.npy").save(arr, length=9, end=1).load(mmap_mode="r+")
print(nt.info())
nt.append(arr).append(arr[0:1])
print(nt.head(3))
print(nt.tail(3))
# del nt

# 文件截断
nt.resize(length=None).load(mmap_mode="r")

nt.resize(length=10).load(mmap_mode="r")
print(nt.array())

nt.resize(length=2).load(mmap_mode="r")
print(nt.array())
print(nt.info())

nt.resize(length=20).load(mmap_mode="r")
print(nt.data())
