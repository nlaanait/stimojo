# ===----------------------------------------------------------------------=== #
# Tests for BitVector and BitMatrix in stimojo/bit_tensor.mojo
# ===----------------------------------------------------------------------=== #

from stimojo.bit_tensor import BitVector, BitMatrix
from testing import assert_equal, assert_true, assert_false, TestSuite
from stimojo import int_type, simd_width


def test_bit_vector_basic():
    print("== test_bit_vector_basic")
    var n = 70
    var bv = BitVector(n)
    assert_equal(bv.n_bits, n)

    bv[0] = True
    bv[63] = True
    bv[64] = True
    bv[69] = True

    assert_true(bv[0])
    assert_true(bv[63])
    assert_true(bv[64])
    assert_true(bv[69])
    assert_false(bv[1])

    var s = String(bv)
    assert_equal(len(s), n)
    assert_equal(s[0], "1")
    assert_equal(s[63], "1")
    assert_equal(s[64], "1")
    assert_equal(s[69], "1")
    assert_equal(s[1], "0")


def test_bit_vector_copy_move():
    print("== test_bit_vector_copy_move")
    var bv = BitVector(10)
    bv[5] = True

    # Copy
    var bv2 = bv
    assert_true(bv2[5])
    bv2[5] = False
    assert_true(bv[5])  # Original unchanged

    # Move
    var bv3 = bv^
    assert_true(bv3[5])


def test_bit_vector_equality():
    print("== test_bit_vector_equality")
    var bv1 = BitVector(10)
    var bv2 = BitVector(10)
    assert_true(bv1 == bv2)
    assert_false(bv1 != bv2)

    bv1[5] = True
    assert_false(bv1 == bv2)
    assert_true(bv1 != bv2)

    bv2[5] = True
    assert_true(bv1 == bv2)


def test_bit_vector_simd():
    print("== test_bit_vector_simd")
    var bv = BitVector(simd_width * 64)
    var val = SIMD[int_type, simd_width](1)
    bv.store[width=simd_width](0, val)

    var loaded = bv.load[width=simd_width](0)
    for i in range(simd_width):
        assert_equal(loaded[i], 1)

    assert_true(bv[0])
    assert_true(bv[65])  # bit 1 of word 1

    var ptr = bv.unsafe_ptr()
    assert_true(ptr)
    assert_equal(ptr.load(0), 1)


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

    var s = String(bm)
    # Check if it contains only 0s and newlines
    assert_true(len(s) > 0)


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


def test_bit_matrix_copy_move():
    print("== test_bit_matrix_copy_move")
    var bm = BitMatrix(2, 2)
    bm[0, 0] = True

    # Copy
    var bm2 = bm
    assert_true(bm2[0, 0])
    bm2[0, 0] = False
    assert_true(bm[0, 0])

    # Move
    var bm3 = bm^
    assert_true(bm3[0, 0])


def test_bit_matrix_equality():
    print("== test_bit_matrix_equality")
    var bm1 = BitMatrix(2, 2)
    var bm2 = BitMatrix(2, 2)
    assert_true(bm1 == bm2)
    assert_false(bm1 != bm2)

    bm1[0, 0] = True
    assert_false(bm1 == bm2)
    assert_true(bm1 != bm2)

    bm2[0, 0] = True
    assert_true(bm1 == bm2)


def test_bit_matrix_transpose():
    print("== test_bit_matrix_transpose")
    var rows = 2
    var cols = 3
    var bm = BitMatrix(rows, cols)
    # 1 0 1
    # 0 1 0
    bm[0, 0] = True
    bm[0, 2] = True
    bm[1, 1] = True

    assert_true(bm.column_major)
    bm.transpose()
    assert_false(bm.column_major)

    # Logical indexing should be preserved
    assert_true(bm[0, 0])
    assert_true(bm[0, 2])
    assert_true(bm[1, 1])
    assert_false(bm[0, 1])
    assert_false(bm[1, 0])
    assert_false(bm[1, 2])

    # Transpose back
    bm.transpose()
    assert_true(bm.column_major)
    assert_true(bm[0, 0])


def test_bit_matrix_xor_col():
    print("== test_bit_matrix_xor_col")
    var bm = BitMatrix(2, 2)
    bm[0, 0] = True
    bm[1, 1] = True
    # 1 0
    # 0 1

    bm.xor_col(1, 0)  # col 1 ^= col 0
    # 1 1
    # 0 1
    assert_true(bm[0, 1])
    assert_true(bm[1, 1])
    assert_true(bm[0, 0])
    assert_false(bm[1, 0])


def test_bit_matrix_word_access():
    print("== test_bit_matrix_word_access")
    var bm = BitMatrix(64, 2)
    bm.set_col_word(0, 0, 1)  # Set first word of col 0 to 1 (bit 0 set)
    assert_true(bm[0, 0])
    assert_equal(bm.col_word(0, 0), 1)

    var ptr = bm.unsafe_ptr()
    assert_true(ptr)


def test_bit_matrix_simd_access():
    print("== test_bit_matrix_simd_access")
    # This tests load_col, store_col, load_row, store_row
    var bm = BitMatrix(simd_width * 64, simd_width * 64)

    # Test col access (should be fast in col_major)
    var v_col = SIMD[int_type, simd_width](7)
    bm.store_col[width=simd_width](5, 0, v_col)
    var l_col = bm.load_col[width=simd_width](5, 0)
    for i in range(simd_width):
        assert_equal(l_col[i], 7)

    # Test row access (triggers transpose)
    var v_row = SIMD[int_type, simd_width](10)
    bm.store_row[width=simd_width](10, 0, v_row)
    assert_false(bm.column_major)
    var l_row = bm.load_row[width=simd_width](10, 0)
    for i in range(simd_width):
        assert_equal(l_row[i], 10)


def test_bit_matrix_simd_tail():
    print("== test_bit_matrix_simd_tail")
    # Choose a size that is not a multiple of simd_width * 64
    # to test tail handling in vectorize.
    var num_words = simd_width * 2 + 1
    var cols = num_words * 64
    var rows = 2
    var bm = BitMatrix(rows, cols)

    # Set some bits in the "tail" word
    var last_col = cols - 1
    bm[0, last_col] = True

    # Perform a SIMD-backed op
    # xor_row will transpose, then vectorize over n_words_per_row
    bm.xor_row(1, 0)

    assert_true(bm[1, last_col])
    assert_true(bm[0, last_col])
    assert_true(bm.is_zero() == False)


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
