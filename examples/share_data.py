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

nt1.append(arr, ringbuffer=False, bulk=False)
print(nt2.data())

nt1.append(arr[0:1], ringbuffer=False, bulk=False)
print(nt3.data())

print("=" * 60)
nt1._test(3, 6)
nt1.data()[:] = 1  # 修改数据成功
print(nt2._raw())

print("=" * 60)
nt1._test(6, 3)
nt1.data()[:] = 2  # 修改数据失败。因为环向数据是拼接的
print(nt2._raw())

del nt1
del nt2
del nt3
os.remove(file)
