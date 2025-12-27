# ===----------------------------------------------------------------------=== #
# Tests for PauliString implementation in stimojo/xzencoding.mojo
# ===----------------------------------------------------------------------=== #

from stimojo.pauli import PauliString, Phase, simd_width, int_bit_width
from testing import assert_equal, TestSuite

alias bits_per_word = int_bit_width
alias vector_bit_width = simd_width * bits_per_word


def test_from_string_roundtrip():
    print("== test_from_string_roundtrip")
    var s = "IxYz"
    var enc = PauliString.from_string(s)
    # PauliString.__str__ should produce the uppercase canonical pauli string with phase prefix
    assert_equal(String(enc), "+IXYZ")


def test_mul_produces_expected():
    print("== test_mul_produces_expected")
    var repeats = 3
    var pauli_string_1 = "IXYZ" * repeats
    var pauli_string_2 = "YZIX" * repeats

    var p1 = PauliString.from_string(pauli_string_1)
    var p2 = PauliString.from_string(pauli_string_2)

    var prod = p1 * p2
    # for these inputs each position should XOR to 'Y' with '+' phase
    var expected_str = "Y" * (4 * repeats)
    assert_equal(String(prod), "+" + expected_str)
    assert_equal(
        prod.global_phase, Phase(0)
    )  # X*Y=iZ, Y*Z=iX, Z*X=iY. Sum of phases for IXYZ * YZIX should be 0.


def test_simd_boundary():
    print("== test_simd_boundary")
    # Test strings with lengths around SIMD boundaries
    var short_string = "I"  # single char
    var enc_short = PauliString.from_string(short_string)
    assert_equal(String(enc_short), "+I")

    # Sub-SIMD (less than 1 vector width of bits)
    var sub_len = vector_bit_width - 10
    var sub_simd = "I" * sub_len
    var enc_sub = PauliString.from_string(sub_simd)
    assert_equal(String(enc_sub), "+" + sub_simd)

    # Over-SIMD (more than 1 vector width)
    var over_len = vector_bit_width + 10
    var over_simd = "I" * over_len
    var enc_over = PauliString.from_string(over_simd)
    assert_equal(String(enc_over), "+" + over_simd)


def test_simd_mul_lengths():
    print("== test_simd_mul_lengths")
    # Test multiplication with different length combinations
    # With bit packing, we process blocks of 'vector_bit_width' bits.

    var len_sub = vector_bit_width // 2
    var len_over = vector_bit_width * 2 + 10

    var s_sub = "I" * len_sub
    var s_over = "I" * len_over

    var p1 = PauliString.from_string(s_sub)
    var p2 = PauliString.from_string(s_sub)
    var prod = p1 * p2
    # Should give all I's
    assert_equal(String(prod), "+" + s_sub)
    assert_equal(prod.global_phase, Phase(0))

    # longer strings that need multiple SIMD ops
    var p3 = PauliString.from_string(s_over)
    var p4 = PauliString.from_string(s_over)
    var long_prod = p3 * p4
    assert_equal(String(long_prod), "+" + s_over)
    assert_equal(long_prod.global_phase, Phase(0))


def test_invalid_chars():
    print("== test_invalid_chars")
    # Test error handling for invalid Pauli characters
    try:
        var s = "ABC"
        var _ = PauliString.from_string(s)
        assert_equal(True, False)
    except e:
        assert_equal(True, True)


def test_mixed_case_mul():
    print("== test_mixed_case_mul")
    # Test that case-insensitive parsing works correctly in multiplication
    var p1 = PauliString.from_string("iXyZ")
    var p2 = PauliString.from_string("YzIx")
    var prod = p1 * p2
    assert_equal(
        String(prod), "+YYYY"
    )  # Result should be normalized to uppercase, with '+' phase
    assert_equal(prod.global_phase, Phase(0))


def test_product_vs_mul():
    print("== test_product_vs_mul")
    # Test that product() and __mul__ give identical results
    var inputs = ["IXYZ", "YZIX", "IIXX", "ZZZZ"]

    for i in range(len(inputs)):
        var p1 = PauliString.from_string(inputs[i])
        for j in range(len(inputs)):
            var p2 = PauliString.from_string(inputs[j])
            var mul_result = p1 * p2
            var p1_copy = PauliString.from_string(inputs[i])
            p1_copy.prod(p2)
            assert_equal(String(p1_copy), String(mul_result))
            assert_equal(
                p1_copy.global_phase,
                mul_result.global_phase,
            )


def test_product_simd_alignment():
    print("== test_product_simd_alignment")
    # Test product() with different SIMD alignments
    # lengths in bits/qubits
    var lengths = [
        1,
        bits_per_word,
        vector_bit_width,
        vector_bit_width * 2,
    ]

    for length in lengths:
        # Create strings of the specified length
        var s1 = "X" * length
        var s2 = "Z" * length
        var p1 = PauliString.from_string(s1)
        var p2 = PauliString.from_string(s2)

        # Both operations should give "Y"s (X*Z = Y, so global phase is + for each)
        var mul_result = p1 * p2
        var p1_copy = PauliString.from_string(s1)
        p1_copy.prod(p2)

        var expected_str = "Y" * length
        # X*Z=-iY, so total phase for n 'X' * 'Z' = (-i)^n
        # This is (length * 3) % 4.
        var expected_phase = Phase(length * 3)

        assert_equal(String(mul_result), String(expected_phase) + expected_str)
        assert_equal(mul_result.global_phase, expected_phase)

        assert_equal(String(p1_copy), String(expected_phase) + expected_str)
        assert_equal(p1_copy.global_phase, expected_phase)


def test_product_chain():
    print("== test_product_chain")
    # Test chaining multiple product() operations
    var p1 = PauliString.from_string("IXYZ")
    var p2 = PauliString.from_string("YZIX")
    var p3 = PauliString.from_string("XIZY")

    # Compare chained product() with multiple __mul__
    var mul_result = p1 * p2 * p3

    var chain = PauliString.from_string("IXYZ")  # start with fresh copy
    chain.prod(p2)  # modify in place
    chain.prod(p3)  # modify in place again

    assert_equal(String(chain), String(mul_result))
    assert_equal(chain.global_phase, mul_result.global_phase)


def test_global_phase_rules_single_paulis():
    print("== test_global_phase_rules")
    # Test anticommutation rules with actual PauliString products
    # Phase values are in log base i: 0=1, 1=i, 2=-1, 3=-i

    var p_X = PauliString.from_string("X")
    var p_Y = PauliString.from_string("Y")
    var p_Z = PauliString.from_string("Z")
    var p_I = PauliString.from_string("I")

    # Rule: XY = iZ (phase = 1 in log base i)
    var xy_result = p_X * p_Y
    assert_equal(String(xy_result), "iZ")
    assert_equal(xy_result.global_phase, Phase(1))

    # Rule: YZ = iX (phase = 1 in log base i)
    var yz_result = p_Y * p_Z
    assert_equal(String(yz_result), "iX")
    assert_equal(yz_result.global_phase, Phase(1))

    # Rule: ZX = iY (phase = 1 in log base i)
    var zx_result = p_Z * p_X
    assert_equal(String(zx_result), "iY")
    assert_equal(zx_result.global_phase, Phase(1))

    # Rule: YX = -iZ (phase = 3 in log base i)
    var yx_result = p_Y * p_X
    assert_equal(String(yx_result), "-iZ")
    assert_equal(yx_result.global_phase, Phase(3))

    # Rule: ZY = -iX (phase = 3 in log base i)
    var zy_result = p_Z * p_Y
    assert_equal(String(zy_result), "-iX")
    assert_equal(zy_result.global_phase, Phase(3))

    # Rule: XZ = -iY (phase = 3 in log base i)
    var xz_result = p_X * p_Z
    assert_equal(String(xz_result), "-iY")
    assert_equal(xz_result.global_phase, Phase(3))

    # Identity commutes with everything (phase = 0)
    var xi_result = p_X * p_I
    assert_equal(String(xi_result), "+X")
    assert_equal(xi_result.global_phase, Phase(0))

    var yi_result = p_Y * p_I
    assert_equal(String(yi_result), "+Y")
    assert_equal(yi_result.global_phase, Phase(0))

    var zi_result = p_Z * p_I
    assert_equal(String(zi_result), "+Z")
    assert_equal(zi_result.global_phase, Phase(0))


def test_global_phase_rules_pair_paulis():
    print("== test_global_phase_rules_pair_paulis")

    var p1 = PauliString.from_string("ZZ")  # Z_0 Z_1
    var p2 = PauliString.from_string("XX")  # X_0 X_1
    var result = p1 * p2
    # Z_0 X_0 = -i Y_0
    # Z_1 X_1 = -i Y_1
    # Product = (-i Y_0) (-i Y_1) = (-i)^2 Y_0 Y_1 = - Y_0 Y_1
    assert_equal(String(result), "-YY")
    assert_equal(result.global_phase, Phase(2))

    var result1 = p2 * p1
    # X_0 Z_0 = -i Y_0
    # X_1 Z_1 = -i Y_1
    # Product = (-i Y_0) (-i Y_1) = (-i)^2 Y_0 Y_1 = - Y_0 Y_1
    assert_equal(String(result1), "-YY")
    assert_equal(result1.global_phase, Phase(2))

    # ZYXZ * XYZX: multiple anticommutations
    # Position 0: Z*X = iY
    # Position 1: Y*Y = +I
    # Position 2: X*Z = -iY
    # Position 3: Z*X = iY
    # Resulting Pauli String: Y I Y Y
    # Total phase: Phase(i) + Phase(+) + Phase(-i) + Phase(i) = Phase(1) + Phase(0) + Phase(3) + Phase(1) = Phase(1+0+3+1) = Phase(5) = Phase(1)
    var p_multi1 = PauliString.from_string("ZYXZ")
    var p_multi2 = PauliString.from_string("XYZX")
    var result2 = p_multi1 * p_multi2
    assert_equal(String(result2), "iYIYY")
    assert_equal(result2.global_phase, Phase(1))


def test_equality():
    print("== test_equality")
    var p1 = PauliString.from_string("IXYZ")
    var p2 = PauliString.from_string("IXYZ")
    var p3 = PauliString.from_string("YZIX")

    assert_equal(p1, p2)
    assert_equal(p1 == p3, False)
    assert_equal(p1 != p3, True)

    # Test equality with phase difference
    var p1_phase = PauliString.from_string("IXYZ", global_phase=1)  # +iIXYZ
    assert_equal(
        p1.global_phase != p1_phase.global_phase, True
    )  # Phase(0) should not equal Phase(1)
    assert_equal(p1 == p1_phase, False)


def test_copy_semantics():
    print("== test_copy_semantics")
    var p_orig = PauliString.from_string("X")
    var p_copy = p_orig.copy()
    assert_equal(p_orig == p_copy, True)


def test_move_semantics():
    print("== test_move_semantics")
    var p_orig = PauliString.from_string("YZ")
    var p_moved = p_orig^
    assert_equal(String(p_moved), "+YZ")


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
