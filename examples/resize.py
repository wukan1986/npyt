import numpy as np

from npyt import NPYT

arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint64)

# 创建文件
nt = NPYT("f3.npy").save(arr, capacity=6).load(mmap_mode="r+")
nt.append(arr, ringbuffer=False)
print(nt._raw())

# 观察取头尾是否正常
print("=" * 60)
print(nt.info())
print(nt.head(3))
print(nt.tail(3))

print("=" * 60)
# del nt

# 文件截断
nt.resize(capacity=None).load(mmap_mode="r")  # 截断到数据区
print(nt._raw())
nt.resize(capacity=2).load(mmap_mode="r")  # 小于数据区，截断到数据区
print(nt._raw())
nt.resize(capacity=8).load(mmap_mode="r")
print(nt._raw())
print("=" * 60)
