from npyt import NPYT

file1 = "tmp1.npy"
# nt1 = NPYT(file1).load(mmap_mode="r")
# print(nt1.info())

with open(file1, "wb") as fp:
    fp.close()

nt1 = NPYT(file1).load(mmap_mode="r")
print(nt1.info())