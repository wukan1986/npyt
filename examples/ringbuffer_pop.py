import numpy as np

from npyt.core import NPYT_RB

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint64)  # + 64

# 创建文件
nt = NPYT_RB("f3.npy").save(arr, capacity=6, end=None).load(mmap_mode="r+")
nt.append(arr, ringbuffer=False)
print(nt.info())
print(nt.data())

print("=" * 60, "取完整")
print(nt.pop(copy=False))
print(nt.info())

print("=" * 60, "取中段")
nt._test(2, 4)
print(nt.pop(copy=False))
print(nt.info())
print(nt.pop(copy=False))
print(nt.info())

print("=" * 60, "环形")
nt._test(4, 2)
print(nt.pop(copy=False))
print(nt.info())
print(nt.pop(copy=False))
print(nt.info())
