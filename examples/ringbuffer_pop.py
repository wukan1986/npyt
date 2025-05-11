import numpy as np

from npyt.core import NPYT_RB

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint64)  # + 64

# 创建文件
nt = NPYT_RB("f3.npy").save(arr, length=6, end=None).load(mmap_mode="r+").append(arr)
print(nt.info())
print(nt.data())

# 测试取完整数
print("=" * 60)
print(nt.pop(copy=False))
print(nt.info())

#  测试取中段
print("=" * 60)
nt._test(2, 4)
print(nt.pop(copy=False))
print(nt.info())
print(nt.pop(copy=False))
print(nt.info())

#  测试环形
print("=" * 60)
nt._test(4, 2)
print(nt.pop(copy=False))
print(nt.info())
print(nt.pop(copy=False))
print(nt.info())
