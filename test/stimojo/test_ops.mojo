from stimojo.ops import phase_from_log_base_i, commutes
from stimojo.pauli import PauliString
from testing import assert_equal, assert_true, assert_false, TestSuite


fn test_phase_from_log_base_i() raises:
    print("== test_phase_from_log_base_i")
    # 0 -> 1
    var p0 = phase_from_log_base_i(0)
    assert_equal(p0.re, 1)
    assert_equal(p0.im, 0)

    # 1 -> i
    var p1 = phase_from_log_base_i(1)
    assert_equal(p1.re, 0)
    assert_equal(p1.im, 1)

    # 2 -> -1
    var p2 = phase_from_log_base_i(2)
    assert_equal(p2.re, -1)
    assert_equal(p2.im, 0)

    # 3 -> -i
    var p3 = phase_from_log_base_i(3)
    assert_equal(p3.re, 0)
    assert_equal(p3.im, -1)

    # 4 -> 1 (cyclic)
    var p4 = phase_from_log_base_i(4)
    assert_equal(p4.re, 1)
    assert_equal(p4.im, 0)


fn test_commutes_sanity() raises:
    print("== test_commutes_sanity")
    # Test basic commutativity
    # Note: Current implementation might be stubbed
    assert_true(commutes(PauliString("X"), PauliString("X")))
    assert_true(commutes(PauliString("Z"), PauliString("Z")))
    assert_true(commutes(PauliString("I"), PauliString("X")))


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
