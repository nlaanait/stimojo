from stimojo.pauli import PauliString
from testing import assert_equal, TestSuite


fn test_copy_semantics() raises:
    print("== test_copy_semantics")
    var p_orig = PauliString(1, "X")
    var p_copy = p_orig.copy()
    assert_equal(p_orig == p_copy, True)


fn test_move_semantics() raises:
    var p_orig = PauliString(1, "YZ")
    var p_moved = p_orig^
    assert_equal(String(p_moved), "+YZ")


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
