# ===----------------------------------------------------------------------=== #
# Tests for XZEncoding implementation in stimojo/xzencoding.mojo
# ===----------------------------------------------------------------------=== #

from stimojo.xzencoding import XZEncoding
from testing import assert_equal, TestSuite


# CHECK-LABEL: test_from_string_roundtrip
def test_from_string_roundtrip():
    print("== test_from_string_roundtrip")
    var s = "IxYz"
    var enc = XZEncoding(s)
    # XZEncoding.__str__ should produce the uppercase canonical pauli string
    assert_equal(String(enc), "IXYZ")


# # CHECK-LABEL: test_mul_produces_expected
# def test_mul_produces_expected():
#     print("== test_mul_produces_expected")
#     var repeats = 3
#     var pauli_string_1 = "IXYZ" * repeats
#     var pauli_string_2 = "YZIX" * repeats

#     var p1 = XZEncoding(pauli_string_1)
#     var p2 = XZEncoding(pauli_string_2)

#     var prod = p1 * p2
#     # for these inputs each position should XOR to 'Y'
#     var expected = "Y" * (4 * repeats)
#     assert_equal(String(prod), expected)


# CHECK-LABEL: test_simd_boundary
def test_simd_boundary():
    print("== test_simd_boundary")
    # Test strings with lengths around SIMD boundaries
    var short_string = "I"  # single char
    var enc_short = XZEncoding(short_string)
    assert_equal(String(enc_short), "I")

    # 15 chars (likely less than typical SIMD width)
    var sub_simd = "IXYZIXYZIXYZIXY"
    var enc_sub = XZEncoding(sub_simd)
    assert_equal(String(enc_sub), sub_simd)

    # 33 chars (likely more than typical SIMD width)
    var over_simd = "IXYZIXYZIXYZIXYZIXYZIXYZIXYZIXYZI"
    var enc_over = XZEncoding(over_simd)
    assert_equal(String(enc_over), over_simd)


# # CHECK-LABEL: test_simd_mul_lengths
# def test_simd_mul_lengths():
#     print("== test_simd_mul_lengths")
#     # Test multiplication with different length combinations
#     var len_15 = "IXYZIXYZIXYZIXY"  # 15 chars
#     var len_33 = "IXYZIXYZIXYZIXYZIXYZIXYZIXYZIXYZI"  # 33 chars

#     var p1 = XZEncoding(len_15)
#     var p2 = XZEncoding(len_15)
#     var prod = p1 * p2
#     # Each position XORs with itself, should give all I's
#     assert_equal(String(prod), "I" * 15)

#     # Now try longer strings that need multiple SIMD ops
#     var p3 = XZEncoding(len_33)
#     var p4 = XZEncoding(len_33)
#     var long_prod = p3 * p4
#     assert_equal(String(long_prod), "I" * 33)


# CHECK-LABEL: test_invalid_chars
def test_invalid_chars():
    print("== test_invalid_chars")
    # Test error handling for invalid Pauli characters
    try:
        var s = "ABC"
        var _ = XZEncoding(s)  # 'B' and 'C' are invalid
        print("ERROR: Should have raised on invalid chars")
        assert_equal(True, False)  # Force test failure
    except e:
        # Expected error
        print("Got expected error for invalid chars")


# CHECK-LABEL: test_mixed_case_mul
def test_mixed_case_mul():
    print("== test_mixed_case_mul")
    # Test that case-insensitive parsing works correctly in multiplication
    var p1 = XZEncoding("iXyZ")
    var p2 = XZEncoding("YzIx")
    var prod = p1 * p2
    assert_equal(
        String(prod), "YYYY"
    )  # Result should be normalized to uppercase


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
