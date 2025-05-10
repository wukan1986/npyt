import numpy as np

from npyt import NPYT

arr = np.array([[1, 2, 3], [4, 5, 6]])

np.save("f4.npy", arr)

nt = NPYT("f4.npy", mode="r").load()
print(nt.array())
print(nt.info())
print(nt.data())
