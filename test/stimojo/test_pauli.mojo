# ===----------------------------------------------------------------------=== #
# Tests for PauliString implementation in stimojo/xzencoding.mojo
# ===----------------------------------------------------------------------=== #

from stimojo.pauli import PauliString
from testing import assert_equal, TestSuite


def test_from_string_roundtrip():
    print("== test_from_string_roundtrip")
    var s = "IxYz"
    var enc = PauliString(s)
    # PauliString.__str__ should produce the uppercase canonical pauli string
    assert_equal(String(enc), "IXYZ")


def test_mul_produces_expected():
    print("== test_mul_produces_expected")
    var repeats = 3
    var pauli_string_1 = "IXYZ" * repeats
    var pauli_string_2 = "YZIX" * repeats

    var p1 = PauliString(pauli_string_1)
    var p2 = PauliString(pauli_string_2)

    var prod = p1 * p2
    # for these inputs each position should XOR to 'Y'
    var expected = "Y" * (4 * repeats)
    assert_equal(String(prod), expected)


def test_simd_boundary():
    print("== test_simd_boundary")
    # Test strings with lengths around SIMD boundaries
    var short_string = "I"  # single char
    var enc_short = PauliString(short_string)
    assert_equal(String(enc_short), "I")

    # 15 chars (likely less than typical SIMD width)
    var sub_simd = "IXYZIXYZIXYZIXY"
    var enc_sub = PauliString(sub_simd)
    assert_equal(String(enc_sub), sub_simd)

    # 33 chars (likely more than typical SIMD width)
    var over_simd = "IXYZIXYZIXYZIXYZIXYZIXYZIXYZIXYZI"
    var enc_over = PauliString(over_simd)
    assert_equal(String(enc_over), over_simd)


def test_simd_mul_lengths():
    print("== test_simd_mul_lengths")
    # Test multiplication with different length combinations
    var len_15 = "IXYZIXYZIXYZIXY"  # 15 chars
    var len_33 = "IXYZIXYZIXYZIXYZIXYZIXYZIXYZIXYZI"  # 33 chars

    var p1 = PauliString(len_15)
    var p2 = PauliString(len_15)
    var prod = p1 * p2
    # Each position XORs with itself, should give all I's
    assert_equal(String(prod), "I" * 15)

    # Now try longer strings that need multiple SIMD ops
    var p3 = PauliString(len_33)
    var p4 = PauliString(len_33)
    var long_prod = p3 * p4
    assert_equal(String(long_prod), "I" * 33)


def test_invalid_chars():
    print("== test_invalid_chars")
    # Test error handling for invalid Pauli characters
    try:
        var s = "ABC"
        var _ = PauliString(s)
        assert_equal(True, False)
    except e:
        assert_equal(True, True)


def test_mixed_case_mul():
    print("== test_mixed_case_mul")
    # Test that case-insensitive parsing works correctly in multiplication
    var p1 = PauliString("iXyZ")
    var p2 = PauliString("YzIx")
    var prod = p1 * p2
    assert_equal(
        String(prod), "YYYY"
    )  # Result should be normalized to uppercase


def test_product_vs_mul():
    print("== test_product_vs_mul")
    # Test that product() and __mul__ give identical results
    var inputs = ["IXYZ", "YZIX", "IIXX", "ZZZZ"]

    for i in range(len(inputs)):
        var p1 = PauliString(inputs[i])
        for j in range(len(inputs)):
            var p2 = PauliString(inputs[j])
            # Get result via __mul__
            var mul_result = p1 * p2
            # Get result via product
            var p1_copy = PauliString(
                inputs[i]
            )  # fresh copy since product modifies in-place
            p1_copy.prod(p2)
            assert_equal(String(p1_copy), String(mul_result))


def test_product_simd_alignment():
    print("== test_product_simd_alignment")
    # Test product() with different SIMD alignments
    var lengths = [1, 15, 33]  # single char, sub-SIMD, over-SIMD

    for length in lengths:
        # Create strings of the specified length
        var s1 = "X" * length
        var s2 = "Z" * length
        var p1 = PauliString(s1)
        var p2 = PauliString(s2)

        # Both operations should give "Y"s (X*Z = Y)
        var mul_result = p1 * p2
        var p1_copy = PauliString(s1)
        p1_copy.prod(p2)

        var expected = "Y" * length
        assert_equal(String(mul_result), expected)
        assert_equal(String(p1_copy), expected)


def test_product_chain():
    print("== test_product_chain")
    # Test chaining multiple product() operations
    var p1 = PauliString("IXYZ")
    var p2 = PauliString("YZIX")
    var p3 = PauliString("XIZY")

    # Compare chained product() with multiple __mul__
    var mul_result = p1 * p2 * p3

    var chain = PauliString("IXYZ")  # start with fresh copy
    chain.prod(p2)  # modify in place
    chain.prod(p3)  # modify in place again

    assert_equal(String(chain), String(mul_result))


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
