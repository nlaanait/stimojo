# ===----------------------------------------------------------------------=== #
# Tests for Tableau operations in stimojo/tableau.mojo
# Mimics @Stim/src/stim/stabilizers/tableau.test.cc
# ===----------------------------------------------------------------------=== #

from stimojo.tableau import Tableau
from stimojo.pauli import PauliString
from testing import assert_equal, TestSuite

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

def test_specialized_ops():
    print("== test_specialized_ops")
    # SWAP
    var t = Tableau(2)
    t.prepend_SWAP(0, 1)
    check_eval(t, "XI", "IX")
    check_eval(t, "IX", "XI")
    check_eval(t, "ZI", "IZ")
    check_eval(t, "IZ", "ZI")

    # H_XY: X<->Y, Z->-Z
    t = Tableau(1)
    t.prepend_H_XY(0)
    check_eval(t, "X", "Y")
    check_eval(t, "Y", "X")
    check_eval(t, "Z", "-Z")

    # H_YZ: Y<->Z, X->-X
    t = Tableau(1)
    t.prepend_H_YZ(0)
    check_eval(t, "Y", "Z")
    check_eval(t, "Z", "Y")
    check_eval(t, "X", "-X")

    # C_XYZ: X->Y, Y->Z, Z->X
    t = Tableau(1)
    t.prepend_C_XYZ(0)
    check_eval(t, "X", "Y")
    check_eval(t, "Y", "Z")
    check_eval(t, "Z", "X")

    # C_ZYX: X->Z, Z->Y, Y->X
    t = Tableau(1)
    t.prepend_C_ZYX(0)
    check_eval(t, "X", "Z")
    check_eval(t, "Z", "Y")
    check_eval(t, "Y", "X")

    # SQRT_XX
    t = Tableau(2)
    t.prepend_SQRT_XX(0, 1)
    # XX -> XX
    # ZI -> YX (anti-commutes with XX, so map to something anti-commuting with XX? )
    # exp(i pi/4 XX) ZI exp(-...) = ZI cos - i ZI XX sin = ZI - i (i Y) X = ZI + YX?
    # Let's check eval
    # SQRT_XX: X_i -> X_i. Z_i -> Z_i X_j (with phase?)
    # Z_0 -> Z_0 X_1 ? No.
    # Stim test for SQRT_XX? Not explicit in snippets.
    # But known property: exp(i pi/4 XX) Z_0 = Y_0 X_1?
    check_eval(t, "XX", "XX")
    check_eval(t, "ZI", "-YX") # Z0 -> -Y0 X1
    check_eval(t, "IZ", "-XY") # Z1 -> -X0 Y1

    # ISWAP
    # X0 -> i Z0 X1 ?
    # ISWAP(X0) = i Z0 X1 ?
    t = Tableau(2)
    t.prepend_ISWAP(0, 1)
    # ISWAP swaps qubits and adds phases.
    # X0 -> i Z0 X1
    # Z0 -> - i X0
    # Wait, ISWAP definitions vary. Stim uses:
    # SWAP * CZ * S * S.
    # SWAP: X0->X1, Z0->Z1.
    # CZ: X1->X1 Z0, X0->X0 Z1.
    # S: X->Y, Z->Z.
    # Let's rely on the fact that prepend_ISWAP is composed of known gates.
    # If it runs without error, it's likely correct given the composition is correct.
    # We can check X0 -> ...
    # X0 ->(S)-> Y0 ->(S)-> Y0 ->(CZ)-> Y0 Z1 ->(SWAP)-> Z0 Y1.
    # Wait, prepend order:
    # prepend_SWAP(q1, q2)
    # prepend_ZCZ(q1, q2)
    # prepend_SQRT_Z(q1)
    # prepend_SQRT_Z(q2)
    # T_new = T_old * ISWAP.
    # Op order applied to state: ISWAP first.
    # Tableau stores $U P U^\dagger$.
    # So if $U = S_1 S_2 CZ SWAP$, then $U X_0 U^\dagger = ...$
    # $SWAP X_0 SWAP = X_1$.
    # $CZ X_1 CZ = X_1 Z_0$.
    # $S_1 S_2 (X_1 Z_0) S^\dagger = (S_2 X_1 S_2^\dagger) (S_1 Z_0 S_1^\dagger) = Y_1 Z_0$.
    # So X0 -> Z0 Y1.
    check_eval(t, "XI", "ZY")
    check_eval(t, "IX", "YZ")

fn main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()