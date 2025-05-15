import os

import numpy as np
import pytest

from npyt import NPYT

file = "tmp.npy"


@pytest.fixture
def arr():
    return np.array([1, 2, 3, 4, 5, 6], dtype=np.uint64)


def test_arr(arr):
    np.save(file, arr)

    nt = NPYT(file).load(mmap_mode="r")
    np.testing.assert_array_equal(arr, nt._raw())
    assert nt.info() is None
    print(nt._raw())
    del nt

    os.remove(file)
