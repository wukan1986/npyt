import numpy as np

from npyt import NPYT

arr = np.array([[1, 2, 3], [4, 5, 6]])

# 创建文件
nt1 = NPYT("f2.npy").save(arr, capacity=10, end=0).load(mmap_mode="r+")
# 只读加载文件
nt2 = NPYT("f2.npy").load(mmap_mode="r")
nt3 = NPYT("f2.npy").load(mmap_mode="r")

nt1.append(arr, ringbuffer=False)
print(nt2.data())

nt1.append(arr[0:1], ringbuffer=False)
print(nt3.data())
print("=" * 60)

nt1._test(3, 6)
nt1.data()[:] = 1  # 修改数据成功
print(nt2._raw())
print("=" * 60)

nt1._test(6, 3)
nt1.data()[:] = 2  # 修改数据失败。因为环向数据是拼接的
print(nt2._raw())
