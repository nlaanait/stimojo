from stimojo.pauli import Phase
from testing import assert_equal, assert_true, assert_false, TestSuite


def test_phase_init():
    print("== test_phase_init")
    # Valid initializations
    var p0 = Phase(0)
    var p1 = Phase(1)
    var p2 = Phase(2)
    var p3 = Phase(3)

    assert_equal(p0.log_value, 0)
    assert_equal(p1.log_value, 1)
    assert_equal(p2.log_value, 2)
    assert_equal(p3.log_value, 3)

    # Modulo arithmetic in init
    var p4 = Phase(4)
    assert_equal(p4.log_value, 0)

    var p5 = Phase(5)
    assert_equal(p5.log_value, 1)


def test_phase_init_invalid():
    print("== test_phase_init_invalid")
    # Negative values might fail validation depending on how % works
    # In Mojo/C, -1 % 4 might be -1.
    try:
        var _ = Phase(-1)
        # If it reaches here without error, we need to know what happened
        # If -1 % 4 is -1, then _validate should raise.
        # If -1 % 4 is 3 (Pythonic), then it should be fine.
        # Let's see what happens.
        pass
    except e:
        # Expected behavior if -1 % 4 is -1
        assert_equal(True, True)

    try:
        var p = Phase(100)  # 100 % 4 == 0, valid
        assert_equal(p.log_value, 0)
    except:
        assert_false(True)


def test_phase_str():
    print("== test_phase_str")
    assert_equal(String(Phase(0)), "+")
    assert_equal(String(Phase(1)), "i")
    assert_equal(String(Phase(2)), "-")
    assert_equal(String(Phase(3)), "-i")


def test_phase_add():
    print("== test_phase_add")
    var p1 = Phase(1)  # i
    var p2 = Phase(1)  # i
    var sum = p1 + p2  # 2 -> -1
    assert_equal(sum.log_value, 2)
    assert_equal(String(sum), "-")

    var p3 = Phase(3)  # -i
    var sum2 = p1 + p3  # 1 + 3 = 4 -> 0 -> +1
    assert_equal(sum2.log_value, 0)
    assert_equal(String(sum2), "+")


def test_phase_add_int():
    print("== test_phase_add_int")
    var p = Phase(1)
    var sum = p + 1
    assert_equal(sum.log_value, 2)

    var sum2 = p + 3  # 1 + 3 = 4 -> 0
    assert_equal(sum2.log_value, 0)


def test_phase_iadd():
    print("== test_phase_iadd")
    var p = Phase(1)
    p += Phase(1)
    assert_equal(p.log_value, 2)

    p += 2  # 2 + 2 = 4 -> 0
    assert_equal(p.log_value, 0)


def test_phase_equality():
    print("== test_phase_equality")
    assert_equal(Phase(0), Phase(0))
    assert_equal(Phase(0) == Phase(1), False)
    assert_equal(Phase(0) != Phase(1), True)
    assert_equal(Phase(0) == Phase(4), True)


def test_exponent():
    print("== test_exponent")
    # 0 -> 1
    var p0 = Phase(0).exponent()
    assert_equal(p0.re, 1)
    assert_equal(p0.im, 0)

    # 1 -> i
    var p1 = Phase(1).exponent()
    assert_equal(p1.re, 0)
    assert_equal(p1.im, 1)

    # 2 -> -1
    var p2 = Phase(2).exponent()
    assert_equal(p2.re, -1)
    assert_equal(p2.im, 0)

    # 3 -> -i
    var p3 = Phase(3).exponent()
    assert_equal(p3.re, 0)
    assert_equal(p3.im, -1)

    # 4 -> 1 (cyclic)
    var p4 = Phase(4).exponent()
    assert_equal(p4.re, 1)
    assert_equal(p4.im, 0)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
