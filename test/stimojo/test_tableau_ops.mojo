# ===----------------------------------------------------------------------=== #
# Tests for Tableau operations in stimojo/tableau.mojo
# Mimics @Stim/src/stim/stabilizers/tableau.test.cc
# ===----------------------------------------------------------------------=== #

from stimojo.tableau import Tableau
from stimojo.pauli import PauliString, Phase
from testing import assert_equal, TestSuite
from collections.list import List


fn check_eval(t: Tableau, input_str: String, expected_full_str: String) raises:
    var p_in_temp = input_str
    var initial_phase_val = 0
    if len(p_in_temp) > 0:
        var first_char = p_in_temp.as_bytes()[0]
        if first_char == ord("-"):
            if len(p_in_temp) > 1 and p_in_temp.as_bytes()[1] == ord(
                "i"
            ):  # Check for "-i"
                initial_phase_val = 3
                p_in_temp = p_in_temp[2:]  # Strip "-i"
            else:  # Just "-"
                initial_phase_val = 2
                p_in_temp = p_in_temp[1:]
        elif first_char == ord("+"):
            initial_phase_val = 0
            p_in_temp = p_in_temp[1:]
        elif first_char == ord("i"):
            initial_phase_val = 1
            p_in_temp = p_in_temp[1:]

    var p_in = PauliString(p_in_temp, global_phase=initial_phase_val)
    var p_out = t.eval(p_in)

    assert_equal(
        String(p_out), expected_full_str
    )  # Compare full string with phase prefix


fn check_apply_within(
    t: Tableau,
    initial_pauli_str: String,
    target_qubits: List[Int],
    expected_full_str: String,
) raises:
    var p_temp_str = initial_pauli_str
    var initial_phase = 0
    if len(p_temp_str) > 0:
        var first_char = p_temp_str.as_bytes()[0]
        if first_char == ord("-"):
            if len(p_temp_str) > 1 and p_temp_str.as_bytes()[1] == ord(
                "i"
            ):  # Check for "-i"
                initial_phase = 3
                p_temp_str = p_temp_str[2:]  # Strip "-i"
            else:  # Just "-"
                initial_phase = 2
                p_temp_str = p_temp_str[1:]
        elif first_char == ord("+"):
            initial_phase = 0
            p_temp_str = p_temp_str[1:]
        elif first_char == ord("i"):
            initial_phase = 1
            p_temp_str = p_temp_str[1:]

    var p = PauliString(p_temp_str, global_phase=initial_phase)

    t.apply_within(p, target_qubits)

    assert_equal(
        String(p), expected_full_str
    )  # Compare full string with phase prefix


def test_eval_gates():
    print("== test_eval_gates")
    # ZCX (CNOT)
    var t = Tableau(2)
    t.prepend_ZCX(0, 1)
    # CNOT(0, 1): X0 -> X0 X1, Z1 -> Z0 Z1
    check_eval(t, "+XI", "+XX")
    check_eval(t, "+IX", "+IX")
    check_eval(t, "+ZI", "+ZI")
    check_eval(t, "+IZ", "+ZZ")
    check_eval(t, "+YY", "-XZ")

    # SQRT_X (0)
    t = Tableau(1)
    t.prepend_SQRT_X(0)
    check_eval(t, "+X", "+X")
    check_eval(t, "+Y", "+Z")
    check_eval(t, "+Z", "-Y")

    # SQRT_Z (S)
    t = Tableau(1)
    t.prepend_SQRT_Z(0)
    check_eval(t, "+X", "+Y")
    check_eval(t, "+Z", "+Z")
    check_eval(t, "+Y", "-X")


def test_eval_y_obs():
    print("== test_eval_y_obs")
    var t = Tableau(1)
    t.prepend_H_XZ(0)
    var py = t.eval_y_obs(0)
    assert_equal(String(py), "-Y")

    t = Tableau(1)
    t.prepend_SQRT_Z(0)
    py = t.eval_y_obs(0)
    assert_equal(String(py), "-X")

    t = Tableau(2)
    t.prepend_ZCX(0, 1)
    py = t.eval_y_obs(1)
    assert_equal(String(py), "+ZY")


def test_direct_h_xy():
    print("== test_direct_h_xy")
    var t = Tableau(1)
    t.prepend_H_XY(0)
    var p_in = PauliString("X")  # This now implies +X
    var p_out = t.eval(p_in)
    assert_equal(String(p_out), "+Y")

    p_in = PauliString("Y")  # This now implies +Y
    p_out = t.eval(p_in)
    assert_equal(String(p_out), "+X")

    p_in = PauliString("Z")  # This now implies +Z
    p_out = t.eval(p_in)
    assert_equal(String(p_out), "-Z")


def test_apply_within():
    print("== test_apply_within")
    var cnot = Tableau(2)
    cnot.prepend_ZCX(0, 1)

    var target_q_01 = List[Int]()
    target_q_01.append(0)
    target_q_01.append(1)

    # Test case 1 from Stim: -XX with CNOT(0,1) -> -XI
    check_apply_within(cnot, "-XX", target_q_01, "-XI")

    # Test case 2 from Stim: +XX with CNOT(0,1) -> +XI
    check_apply_within(cnot, "+XX", target_q_01, "+XI")

    # Test case from previous example: XZI on a 3-qubit Pauli string acted on by a 2-qubit CNOT(0,1) tableau
    # Initial: XZI, phase 0. Tableau: CNOT(0,1). Target qubits: 0, 1.
    # Expected: YYI, phase 2.

    var t_cnot_2q = Tableau(2)
    t_cnot_2q.apply_CX(0, 1)

    var target_q_01_3_qubit_context = List[Int]()
    target_q_01_3_qubit_context.append(0)
    target_q_01_3_qubit_context.append(1)

    # We pass the CNOT tableau, the initial XZI (3-qubit), target qubits 0,1, and expected result.
    check_apply_within(t_cnot_2q, "+XZI", target_q_01_3_qubit_context, "-YYI")


def test_call_operator():
    print("== test_call_operator")
    var t = Tableau(1)
    t.prepend_H_XZ(0)  # H gate: X->Z, Z->X

    var p_x = PauliString("X")
    var res_x = t(p_x)
    assert_equal(String(res_x), "+Z")

    var p_z = PauliString("Z")
    var res_z = t(p_z)
    assert_equal(String(res_z), "+X")


def test_call_operator_expanded():
    print("== test_call_operator_expanded")
    # Test for Tableau(1) and various gates
    var t = Tableau(1)

    # X gate
    t.apply_X(0)
    check_eval(t, "+Z", "-Z")
    check_eval(t, "+X", "+X")

    # Y gate
    t = Tableau(1)  # Reset tableau
    t.apply_Y(0)
    check_eval(t, "+X", "-X")
    check_eval(t, "+Z", "-Z")

    # S gate
    t = Tableau(1)  # Reset tableau
    t.apply_S(0)
    check_eval(t, "+X", "+Y")
    check_eval(t, "+Z", "+Z")

    # H gate
    t = Tableau(1)  # Reset tableau
    t.apply_hadamard(0)
    check_eval(t, "+X", "+Z")
    check_eval(t, "+Z", "+X")

    # Test for Tableau(2) and CX gate
    t = Tableau(2)  # Reset tableau
    t.apply_CX(0, 1)
    check_eval(t, "+XI", "+XX")
    check_eval(t, "+IX", "+IX")
    check_eval(t, "+ZI", "+ZI")
    check_eval(t, "+IZ", "+ZZ")

    # Test with non-zero initial global phase
    check_eval(t, "iIX", "iIX")


def test_call_and_apply_within_equivalence():
    print("== test_call_and_apply_within_equivalence")
    var t_n_qubits = 2  # Tableau size
    var p_n_qubits = 3  # PauliString size

    # Case 1: Tableau size matches PauliString size, full conjugation
    var t_cx = Tableau(t_n_qubits)
    t_cx.apply_CX(0, 1)  # CNOT on 0,1

    var p_orig = PauliString("IX", global_phase=1)  # Initial +iIX
    var expected_full_str = "iIX"  # CNOT on IX -> IX, so phase should remain i

    # Using __call__ (out-of-place)
    var p_called = t_cx(p_orig)
    assert_equal(String(p_called), expected_full_str)

    # Using apply_within (in-place)
    var p_within = p_orig.copy()
    var full_target_qubits = List[Int]()
    for i in range(t_n_qubits):
        full_target_qubits.append(i)
    t_cx.apply_within(p_within, full_target_qubits)

    assert_equal(String(p_within), expected_full_str)
    assert_equal(String(p_called), String(p_within))

    # Case 2: Tableau applied to a subsystem of a larger PauliString
    t_n_qubits = 1
    p_n_qubits = 3
    var t_h = Tableau(t_n_qubits)
    t_h.apply_hadamard(0)  # H on 0

    var p_orig_large = PauliString("XIZ")  # +XIZ
    var expected_full_str_large = "+ZIZ"  # H on X -> Z, so +ZIZ

    # Using apply_within on a subsystem (qubit 0)
    var p_within_large = p_orig_large.copy()
    var target_qubits_sub = List[Int]()
    target_qubits_sub.append(0)  # Apply to qubit 0
    t_h.apply_within(p_within_large, target_qubits_sub)
    assert_equal(String(p_within_large), expected_full_str_large)

    # Simulating __call__ behavior for a subsystem for comparison
    var p_subsystem_orig = PauliString("X")  # Subsystem X from XIZ
    var p_subsystem_called = t_h(p_subsystem_orig)  # H(X) = Z

    # Construct the expected larger string
    var p_expected_large = p_orig_large.copy()
    p_expected_large.pauli_string = (
        String(p_subsystem_called) + "IZ"
    )  # Z on first qubit, I on second, Z on third
    p_expected_large.global_phase = p_subsystem_called.global_phase

    assert_equal(String(p_within_large), "+ZIZ")

    # Demonstrate that direct __call__ on mismatched sizes raises an error (as implemented)
    var t_small = Tableau(1)
    try:
        var p_mismatched = PauliString("XX")
        var res = t_small(p_mismatched)
        assert_equal(True, False)  # Should not reach here
    except e:
        assert_equal(True, True)  # Expected error caught


def test_elementary_ops():
    print("== test_elementary_ops")
    # X conjugation
    var t_x = Tableau(1)
    t_x.apply_X(0)
    check_eval(t_x, "+Z", "-Z")
    check_eval(t_x, "+X", "+X")

    # Y conjugation
    var t_y = Tableau(1)
    t_y.apply_Y(0)
    check_eval(t_y, "+X", "-X")
    check_eval(t_y, "+Z", "-Z")

    # Z conjugation
    var t_z = Tableau(1)
    t_z.apply_Z(0)
    check_eval(t_z, "+Z", "+Z")
    check_eval(t_z, "+X", "-X")

    # S conjugation
    var t_s = Tableau(1)
    t_s.apply_S(0)
    check_eval(t_s, "+X", "+Y")
    check_eval(t_s, "+Z", "+Z")

    # H conjugation
    var t_h = Tableau(1)
    t_h.apply_hadamard(0)
    check_eval(t_h, "+X", "+Z")
    check_eval(t_h, "+Z", "+X")

    # CX conjugation
    var t_cx = Tableau(2)
    t_cx.apply_CX(0, 1)
    check_eval(t_cx, "+XI", "+XX")
    check_eval(t_cx, "+IX", "+IX")
    check_eval(t_cx, "+ZI", "+ZI")
    check_eval(t_cx, "+IZ", "+ZZ")


fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
