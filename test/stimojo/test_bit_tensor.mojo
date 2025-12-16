# ===----------------------------------------------------------------------=== #
# Tests for BitMatrix in stimojo/bit_tensor.mojo
# ===----------------------------------------------------------------------=== #

from stimojo.bit_tensor import BitMatrix
from testing import assert_equal, assert_true, assert_false, TestSuite


def test_bit_matrix_init_zero():
    print("== test_bit_matrix_init_zero")
    var rows = 4
    var cols = 128
    var bm = BitMatrix(rows, cols)

    assert_true(bm.is_zero())
    assert_false(bm.is_identity())

    for r in range(rows):
        for c in range(cols):
            assert_false(bm[r, c])


def test_bit_matrix_set_get():
    print("== test_bit_matrix_set_get")
    var bm = BitMatrix(2, 64)

    bm[0, 0] = True
    bm[1, 63] = True

    assert_true(bm[0, 0])
    assert_false(bm[0, 1])
    assert_false(bm[1, 0])
    assert_true(bm[1, 63])

    assert_false(bm.is_zero())


def test_bit_matrix_identity():
    print("== test_bit_matrix_identity")
    var n = 64
    var bm = BitMatrix(n, n)

    # Manually set diagonal
    for i in range(n):
        bm[i, i] = True

    assert_true(bm.is_identity())
    assert_false(bm.is_zero())

    # Perturb
    bm[0, 1] = True
    assert_false(bm.is_identity())

    # Fix back
    bm[0, 1] = False
    assert_true(bm.is_identity())

    # Unset diagonal
    bm[0, 0] = False
    assert_false(bm.is_identity())


def test_swap_rows():
    print("== test_swap_rows")
    var bm = BitMatrix(2, 64)
    bm[0, 0] = True  # Row 0: 100...
    bm[1, 1] = True  # Row 1: 010...

    bm.swap_rows(0, 1)

    assert_false(bm[0, 0])
    assert_true(bm[0, 1])  # Row 0 is now old Row 1
    assert_true(bm[1, 0])  # Row 1 is now old Row 0
    assert_false(bm[1, 1])


def test_xor_row():
    print("== test_xor_row")
    var bm = BitMatrix(2, 64)
    # Row 0: 100...
    # Row 1: 010...
    bm[0, 0] = True
    bm[1, 1] = True

    # Row 0 ^= Row 1 -> 110...
    bm.xor_row(0, 1)

    assert_true(bm[0, 0])
    assert_true(bm[0, 1])
    assert_false(bm[0, 2])

    # Row 1 unchanged
    assert_false(bm[1, 0])
    assert_true(bm[1, 1])

    # XOR again: Row 0 ^= Row 1 -> 100...
    bm.xor_row(0, 1)
    assert_true(bm[0, 0])
    assert_false(bm[0, 1])


def test_is_zero_large():
    print("== test_is_zero_large")
    var bm = BitMatrix(10, 1000)
    assert_true(bm.is_zero())

    bm[5, 500] = True
    assert_false(bm.is_zero())

    bm[5, 500] = False
    assert_true(bm.is_zero())


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
