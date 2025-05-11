import numpy as np

from npyt import NPYT

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint64)  # + 64
a = arr[0:3] * 10

# 创建文件
nt = NPYT("f3.npy").save(arr, capacity=4, end=None).load(mmap_mode="r+")
print(nt.info())
print(nt._raw())

print("=" * 60, "满数据，添加失败")
nt._test(0, 4)
print(nt.append(a[:1], ringbuffer=True))
nt._test(1, 0)
print(nt.append(a[:1], ringbuffer=True))

print("=" * 60, "后段一个空位，添加")
nt._test(2, 3)
print(nt.append(a, ringbuffer=False))
print(nt._raw())
print(nt.append(a[-1:], ringbuffer=False))
print(nt._raw())
print(nt.append(a[-1:], ringbuffer=True))
print(nt._raw())

print("=" * 60, "后段2个空位，添加两行")
nt._test(2, 2)
print(nt.append(a, ringbuffer=False))
print(nt._raw())
print(nt.append(a[-1:], ringbuffer=False))
print(nt._raw())
print(nt.append(a[-1:], ringbuffer=True))
print(nt._raw())
