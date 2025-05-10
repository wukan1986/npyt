import numpy as np

from npyt import NPYT

arr = np.array([[1, 2, 3], [4, 5, 6]])

# 创建文件
nt1 = NPYT("f2.npy").save(arr, length=10, end=0).load(mmap_mode="r+")
# 只读加载文件
nt2 = NPYT("f2.npy").load(mmap_mode="r")
nt3 = NPYT("f2.npy").load(mmap_mode="r")

nt1.append(arr)
print(nt2.data())

nt1.append(arr[0:1])
print(nt3.data())
