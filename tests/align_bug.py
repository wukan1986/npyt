import numpy as np
from numba import typeof
from numba.experimental import jitclass

# _origin_descr_to_dtype = np.lib.format.descr_to_dtype
#
#
# def dtype_with_align(dtype: np.dtype):
#     if dtype.names is None:
#         return dtype
#     align = len(dtype.descr) > len(dtype.names)
#     descr = [(x, y) for x, y in dtype.descr if x != '']
#     return np.dtype(descr, align=align)
#
#
# def _descr_to_dtype(descr):
#     return dtype_with_align(_origin_descr_to_dtype(descr))
#
# np.lib.format.descr_to_dtype = _descr_to_dtype

file = "tmp.npy"

dtype = np.dtype([
    ("a", np.int32),
    ("b", np.float32),
    ("c", np.bool_),
    ("d", "U9"),
], align=True)

arr1 = np.array([
    (1, 2.0, True, "hello"),
    (2, 3.0, False, "world")
], dtype=dtype)


@jitclass(spec=[('arr', typeof(np.empty(1, dtype=dtype)))])
class Test:
    def __init__(self, arr: np.ndarray):
        self.arr = arr

    def modify(self, arr: np.ndarray):
        self.arr[:] = arr


np.save(file, arr1)
arr2 = np.load(file, mmap_mode='r+')  # missing align
print(arr1.dtype)  # align=True
print(arr2.dtype)  # align=False
arr2[:] = arr1  # OK

t = Test(arr2)
t.modify(arr1)  # assert fromty.mutable != toty.mutable or toty.layout == 'A'
