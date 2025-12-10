# ===----------------------------------------------------------------------=== #
# Tests for Tableau implementation in stimojo/tableau.mojo
# ===----------------------------------------------------------------------=== #

from stimojo.tableau import Tableau
from testing import assert_equal, TestSuite


def test_tableau_init_identity():
    print("== test_tableau_init_identity")
    var n_qubits = 3
    var t = Tableau(n_qubits)

    assert_equal(t.n_qubits, n_qubits)

    # Check identity structure
    for k in range(n_qubits):
        for q in range(n_qubits):
            # Check X output for k-th generator
            if q == k:
                assert_equal(t.x_out_x(k, q), 1)
                assert_equal(t.z_out_z(k, q), 1)
            else:
                assert_equal(t.x_out_x(k, q), 0)
                assert_equal(t.z_out_z(k, q), 0)

            # Z components of X output should be 0
            assert_equal(t.x_out_z(k, q), 0)
            # X components of Z output should be 0
            assert_equal(t.z_out_x(k, q), 0)

        # Signs should be positive
        assert_equal(t.x_sign(k), 0)
        assert_equal(t.z_sign(k), 0)


def test_tableau_copy():
    print("== test_tableau_copy")
    var n_qubits = 2
    var t1 = Tableau(n_qubits)

    # Modify t1
    t1.set_x_out_x(0, 0, 0)
    t1.set_x_sign(0, 1)

    var t2 = t1.copy()

    # Check t2 has same values
    assert_equal(t2.x_out_x(0, 0), 0)
    assert_equal(t2.x_sign(0), 1)

    # Modify t2
    t2.set_x_out_x(0, 0, 1)
    t2.set_x_sign(0, 0)

    # Ensure t1 unchanged
    assert_equal(t1.x_out_x(0, 0), 0)
    assert_equal(t1.x_sign(0), 1)

    # Ensure t2 changed
    assert_equal(t2.x_out_x(0, 0), 1)
    assert_equal(t2.x_sign(0), 0)


def test_tableau_apply_hadamard():
    print("== test_tableau_apply_hadamard")
    var n = 1
    var t = Tableau(n)  # Identity tableau

    # Initial state (Identity)
    # X0 -> +X0
    # Z0 -> +Z0
    assert_equal(t.x_out_x(0, 0), 1)
    assert_equal(t.x_out_z(0, 0), 0)
    assert_equal(t.z_out_x(0, 0), 0)
    assert_equal(t.z_out_z(0, 0), 1)
    assert_equal(t.x_sign(0), 0)
    assert_equal(t.z_sign(0), 0)

    t.apply_hadamard(0)  # Apply H on qubit 0

    # Expected state after H0:
    # X0 -> +Z0
    # Z0 -> +X0
    assert_equal(t.x_out_x(0, 0), 0)
    assert_equal(t.x_out_z(0, 0), 1)
    assert_equal(t.z_out_x(0, 0), 1)
    assert_equal(t.z_out_z(0, 0), 0)
    assert_equal(t.x_sign(0), 0)
    assert_equal(t.z_sign(0), 0)

    # Test with Y operator to check sign flip
    n = 1
    t = Tableau(n)
    # Manually set tableau to X0 -> +Y0, Z0 -> +Z0
    t.set_x_out_x(0, 0, 1)
    t.set_x_out_z(0, 0, 1)  # This makes it Y0

    assert_equal(t.x_out_x(0, 0), 1)
    assert_equal(t.x_out_z(0, 0), 1)
    assert_equal(t.x_sign(0), 0)  # Should be +Y0

    t.apply_hadamard(0)

    # H Y0 H = -Y0
    assert_equal(t.x_out_x(0, 0), 1)
    assert_equal(t.x_out_z(0, 0), 1)  # Still Y0
    assert_equal(t.x_sign(0), 1)  # Sign should be flipped to -


def test_single_qubit_gates():
    print("== test_single_qubit_gates")
    # Testing X, Y, Z, S
    var n = 1

    # X: X -> X, Z -> -Z
    var t = Tableau(n)
    t.apply_X(0)
    assert_equal(t.x_out_x(0, 0), 1)  # X0
    assert_equal(t.z_out_z(0, 0), 1)  # Z0
    assert_equal(t.z_sign(0), 1)  # Z0 sign flipped (-)
    assert_equal(t.x_sign(0), 0)  # X0 sign same (+)

    # Y: X -> -X, Z -> -Z
    t = Tableau(n)
    t.apply_Y(0)
    assert_equal(t.x_out_x(0, 0), 1)  # X0
    assert_equal(t.x_sign(0), 1)  # X0 sign flipped (-)
    assert_equal(t.z_out_z(0, 0), 1)  # Z0
    assert_equal(t.z_sign(0), 1)  # Z0 sign flipped (-)

    # Z: X -> -X, Z -> Z
    t = Tableau(n)
    t.apply_Z(0)
    assert_equal(t.x_out_x(0, 0), 1)  # X0
    assert_equal(t.x_sign(0), 1)  # X0 sign flipped (-)
    assert_equal(t.z_out_z(0, 0), 1)  # Z0
    assert_equal(t.z_sign(0), 0)  # Z0 sign same (+)

    # S: X -> Y, Z -> Z
    t = Tableau(n)
    t.apply_S(0)
    # X0 -> Y0 (+Y)
    assert_equal(t.x_out_x(0, 0), 1)
    assert_equal(t.x_out_z(0, 0), 1)
    assert_equal(t.x_sign(0), 0)
    # Z0 -> Z0 (+Z)
    assert_equal(t.z_out_z(0, 0), 1)
    assert_equal(t.z_out_x(0, 0), 0)
    assert_equal(t.z_sign(0), 0)

    # S_dag: X -> -Y, Z -> Z
    t = Tableau(n)
    t.apply_S_dag(0)
    # X0 -> -Y0
    assert_equal(t.x_out_x(0, 0), 1)
    assert_equal(t.x_out_z(0, 0), 1)
    assert_equal(t.x_sign(0), 1)  # Sign flipped
    # Z0 -> Z0
    assert_equal(t.z_out_z(0, 0), 1)
    assert_equal(t.z_sign(0), 0)


def test_two_qubit_gates():
    print("== test_two_qubit_gates")
    var n = 2

    # CX (0 -> 1)
    # X0 -> X0 X1, X1 -> X1
    # Z0 -> Z0, Z1 -> Z0 Z1
    var t = Tableau(n)
    t.apply_CX(0, 1)

    # X0 -> X0 X1
    assert_equal(t.x_out_x(0, 0), 1)
    assert_equal(t.x_out_x(0, 1), 1)  # X on target
    assert_equal(t.x_sign(0), 0)

    # Z1 -> Z0 Z1
    assert_equal(t.z_out_z(1, 1), 1)
    assert_equal(t.z_out_z(1, 0), 1)  # Z on control
    assert_equal(t.z_sign(1), 0)

    # CZ (0, 1)
    # X0 -> X0 Z1, X1 -> Z0 X1
    # Z -> Z
    t = Tableau(n)
    t.apply_CZ(0, 1)

    # X0 -> X0 Z1
    assert_equal(t.x_out_x(0, 0), 1)
    assert_equal(t.x_out_z(0, 1), 1)  # Z on target

    # X1 -> Z0 X1
    assert_equal(t.x_out_x(1, 1), 1)
    assert_equal(t.x_out_z(1, 0), 1)  # Z on control

    # CY (0, 1)
    # X0 -> X0 Y1, X1 -> Z0 X1
    # Z0 -> Z0, Z1 -> Z0 Z1
    # (Derived from decomposition: CY = S_dag(t) CX(c, t) S(t))
    t = Tableau(n)
    t.apply_CY(0, 1)

    # X0 -> X0 Y1 (X0 X1 Z1)
    assert_equal(t.x_out_x(0, 0), 1)
    assert_equal(t.x_out_x(0, 1), 1)
    assert_equal(t.x_out_z(0, 1), 1)

    # Z1 -> Z0 Z1
    assert_equal(t.z_out_z(1, 0), 1)
    assert_equal(t.z_out_z(1, 1), 1)


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
