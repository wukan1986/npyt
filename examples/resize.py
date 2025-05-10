import numpy as np

from npyt import NPYT

arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64) + 64

# 创建文件
nt = NPYT("f3.npy", max_length=9, mode="w+").save(arr, end=1).load()
print(nt.info())
nt.append(arr).append(arr[0:1])
print(nt.head(3))
print(nt.tail(3))
# del nt

# 文件截断
nt.resize(length=None).load()

nt.resize(length=10).load()
print(nt.array())

nt.resize(length=2).load()
print(nt.array())
print(nt.info())

nt.resize(length=20).load()
print(nt.data())
