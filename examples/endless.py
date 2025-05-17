import numpy as np

from npyt import NPY8

if __name__ == '__main__':
    ns = NPY8('demo', 21, 8, dtype=np.int32).load()

    for i in range(9):
        ns.append(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + i * 10)

    for i in range(6):
        print(ns.read(1000))
    print(ns.tail(50))
    print(ns.head(50))

    for i in range(9, 15):
        ns.append(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + i * 10)

    for i in range(5):
        print(ns.read(1000))
    print(ns.tail(100))

    for i in range(15, 40):
        ns.append(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + i * 10)

    ns.merge(4)

    ns.remove()
