from stimojo.pauli import XZEncoding, BitVector, int_type
from testing import assert_equal, assert_true, assert_false, TestSuite
from memory import UnsafePointer


def test_xz_encoding_init():
    print("== test_xz_encoding_init")
    var n = 10
    var enc = XZEncoding(n)

    assert_equal(enc.n_qubits, n)
    # Should default to all zeros (Identity)
    assert_equal(String(enc), "I" * n)

    for i in range(n):
        var val = enc[i]
        assert_equal(val[0], 0)
        assert_equal(val[1], 0)


def test_xz_encoding_set_get():
    print("== test_xz_encoding_set_get")
    var enc = XZEncoding(4)
    # I I I I

    # Set 0 to X
    enc[0] = (1, 0)
    # Set 1 to Z
    enc[1] = (0, 1)
    # Set 2 to Y
    enc[2] = (1, 1)
    # Set 3 to I (explicit)
    enc[3] = (0, 0)

    assert_equal(String(enc), "XZYI")

    var v0 = enc[0]
    assert_equal(v0[0], 1)
    assert_equal(v0[1], 0)

    var v1 = enc[1]
    assert_equal(v1[0], 0)
    assert_equal(v1[1], 1)

    var v2 = enc[2]
    assert_equal(v2[0], 1)
    assert_equal(v2[1], 1)


def test_xz_encoding_equality():
    print("== test_xz_encoding_equality")

    var enc1 = XZEncoding(4)
    var enc2 = XZEncoding(4)

    assert_equal(enc1, enc2)

    enc1[0] = (1, 0)  # X

    assert_false(enc1 == enc2)

    assert_true(enc1 != enc2)

    enc2[0] = (1, 0)  # X

    assert_equal(enc1, enc2)


def test_random_encoding():
    print("== test_random_encoding")

    var n = 100

    var enc = XZEncoding.random_encoding(n)

    # Very unlikely to be all identity for large n

    assert_false(String(enc) == "I" * n)

    # Check bounds

    assert_equal(enc.n_qubits, n)


def test_copy_move():
    print("== test_copy_move")

    var enc1 = XZEncoding(2)

    enc1[0] = (1, 1)  # Y

    var enc2 = enc1  # Copy

    assert_equal(String(enc2), "YI")

    assert_equal(enc1, enc2)

    enc2[1] = (0, 1)  # Z

    assert_equal(String(enc2), "YZ")

    assert_equal(String(enc1), "YI")  # enc1 should not change

    var enc3 = enc2^  # Move
    assert_equal(String(enc3), "YZ")


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
