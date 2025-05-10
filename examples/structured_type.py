import numpy as np

from npyt import NPYT

dtype = np.dtype([
    ("a", np.int32),
    ("b", np.float32),
    ("c", np.bool_),
    ("d", "U9"),
], align=True)
print(dtype)

arr = np.array([
    (1, 2.0, True, "hello"),
    (2, 3.0, False, "world")
], dtype=dtype)
print(arr.shape)

nt = NPYT("f1.npy", ).save(arr, length=9).load(mmap_mode="r+")
nt.append(arr)
nt.append(arr[0:1])
print(nt.array())

# 用numpy原生函数也可以读
print(np.load("f1.npy"))
