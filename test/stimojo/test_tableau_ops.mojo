# ===----------------------------------------------------------------------=== #
# Tests for Tableau operations in stimojo/tableau.mojo
# Mimics @Stim/src/stim/stabilizers/tableau.test.cc
# ===----------------------------------------------------------------------=== #

from stimojo.tableau import Tableau
from stimojo.pauli import PauliString
from testing import assert_equal, TestSuite
from collections.list import List

fn check_eval(t: Tableau, input_str: String, expected_str: String) raises:
    var p_in = PauliString(input_str)
    var p_out = t.eval(p_in)
    
    var expected_phase = 0
    var core_str = expected_str
    if len(expected_str) > 0:
        var first_char = expected_str.as_bytes()[0]
        if first_char == ord("-"):
            expected_phase = 2
            core_str = expected_str[1:]
        elif first_char == ord("+"):
            expected_phase = 0
            core_str = expected_str[1:]
            
    assert_equal(String(p_out), core_str)
    assert_equal(p_out.global_phase, expected_phase)

fn check_apply_within(t: Tableau, initial_pauli_str: String, target_qubits: List[Int], expected_pauli_str: String) raises:
    var p_temp_str = initial_pauli_str
    var initial_phase = 0
    if len(p_temp_str) > 0:
        var first_char = p_temp_str.as_bytes()[0]
        if first_char == ord("-"):
            initial_phase = 2
            p_temp_str = p_temp_str[1:]
        elif first_char == ord("+"):
            initial_phase = 0
            p_temp_str = p_temp_str[1:]
            
    var p = PauliString(p_temp_str, global_phase = initial_phase)
    
    var expected_phase = 0
    var core_str = expected_pauli_str
    if len(expected_pauli_str) > 0:
        var first_char = expected_pauli_str.as_bytes()[0]
        if first_char == ord("-"):
            expected_phase = 2
            core_str = expected_pauli_str[1:]
        elif first_char == ord("+"):
            expected_phase = 0
            core_str = expected_pauli_str[1:]
            
    t.apply_within(p, target_qubits)
    
    assert_equal(String(p), core_str)
    assert_equal(p.global_phase, expected_phase)

def test_eval_gates():
    print("== test_eval_gates")
    # ZCX (CNOT)
    var t = Tableau(2)
    t.prepend_ZCX(0, 1)
    # CNOT(0, 1): X0 -> X0 X1, Z1 -> Z0 Z1
    check_eval(t, "XI", "XX")
    check_eval(t, "IX", "IX") 
    check_eval(t, "ZI", "ZI")
    check_eval(t, "IZ", "ZZ")
    check_eval(t, "YY", "-XZ")

    # SQRT_X (0)
    t = Tableau(1)
    t.prepend_SQRT_X(0)
    check_eval(t, "X", "X")
    check_eval(t, "Y", "Z")
    check_eval(t, "Z", "-Y")

    # SQRT_Z (S)
    t = Tableau(1)
    t.prepend_SQRT_Z(0)
    check_eval(t, "X", "Y")
    check_eval(t, "Z", "Z")
    check_eval(t, "Y", "-X")

def test_eval_y_obs():
    print("== test_eval_y_obs")
    var t = Tableau(1)
    t.prepend_H_XZ(0)
    var py = t.eval_y_obs(0)
    assert_equal(String(py), "Y")
    assert_equal(py.global_phase, 2) # -Y

    t = Tableau(1)
    t.prepend_SQRT_Z(0)
    py = t.eval_y_obs(0)
    assert_equal(String(py), "X")
    assert_equal(py.global_phase, 2) # -X

    t = Tableau(2)
    t.prepend_ZCX(0, 1)
    py = t.eval_y_obs(1)
    assert_equal(String(py), "ZY")
    assert_equal(py.global_phase, 0)

def test_direct_h_xy():
    print("== test_direct_h_xy")
    var t = Tableau(1)
    t.prepend_H_XY(0)
    var p_in = PauliString("X")
    var p_out = t.eval(p_in)
    assert_equal(String(p_out), "Y")
    assert_equal(p_out.global_phase, 0)
    
    p_in = PauliString("Y")
    p_out = t.eval(p_in)
    assert_equal(String(p_out), "X")
    assert_equal(p_out.global_phase, 0)
    
    p_in = PauliString("Z")
    p_out = t.eval(p_in)
    assert_equal(String(p_out), "Z")
    assert_equal(p_out.global_phase, 2) # -Z

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
    # The `apply_within` function in Mojo handles `target_qubits` that are a subset of the PauliString's qubits.
    var t_cnot_2q = Tableau(2)
    t_cnot_2q.apply_CX(0, 1) # This creates a 2-qubit CNOT tableau
    
    var target_q_01_3_qubit_context = List[Int]()
    target_q_01_3_qubit_context.append(0)
    target_q_01_3_qubit_context.append(1)
    
    # We pass the CNOT tableau, the initial XZI (3-qubit), target qubits 0,1, and expected result.
    check_apply_within(t_cnot_2q, "XZI", target_q_01_3_qubit_context, "-YYI") # -YYI = YYI with phase 2

def test_call_operator():
    print("== test_call_operator")
    var t = Tableau(1)
    t.prepend_H_XZ(0) # H gate: X->Z, Z->X
    
    var p_x = PauliString("X")
    var res_x = t(p_x)
    assert_equal(String(res_x), "Z")
    
    var p_z = PauliString("Z")
    var res_z = t(p_z)
    assert_equal(String(res_z), "X")

fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()