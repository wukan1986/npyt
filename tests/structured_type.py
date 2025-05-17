import os

import numpy as np
import pytest

from npyt import NPYT

file = "tmp.npy"


@pytest.fixture
def some_data():
    dtype = np.dtype([
        ("a", np.int32),
        ("b", np.float32),
        ("c", np.bool_),
        ("d", "U9"),
    ], align=True)

    arr = np.array([
        (1, 2.0, True, "hello"),
        (2, 3.0, False, "world")
    ], dtype=dtype)

    yield arr
    os.remove(file)


def test_some_data(some_data):
    nt = NPYT(file).save(some_data, capacity=5, skip_if_exists=False).load(mmap_mode="r+")
    nt.append(some_data)
    nt.append(some_data[0:1])

    # 用numpy原生函数也可以读
    np.testing.assert_array_equal(np.load(file), nt.data())
