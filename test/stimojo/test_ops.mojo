from stimojo.ops import commutes
from stimojo.pauli import PauliString
from testing import assert_equal, assert_true, assert_false, TestSuite


fn test_commutes_sanity() raises:
    print("== test_commutes_sanity")
    # Test basic commutativity
    # Note: Current implementation might be stubbed
    assert_true(
        commutes(PauliString.from_string("X"), PauliString.from_string("X"))
    )
    assert_true(
        commutes(PauliString.from_string("Z"), PauliString.from_string("Z"))
    )
    assert_true(
        commutes(PauliString.from_string("I"), PauliString.from_string("X"))
    )


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
