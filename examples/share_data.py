import numpy as np

from npyt import NPYT

arr = np.array([[1, 2, 3], [4, 5, 6]])

# 创建文件
nt1 = NPYT("f2.npy", max_length=9, mode="w+").save(arr, end=0).load()
nt2 = NPYT("f2.npy", mode="r").load()
nt3 = NPYT("f2.npy", mode="r").load()

nt1.append(arr)
print(nt2.data())

nt1.append(arr[0:1])
print(nt3.data())
