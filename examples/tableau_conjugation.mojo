from stimojo.tableau import Tableau
from stimojo.pauli import PauliString
from collections.list import List


fn main() raises:
    print("---------------------------------------------------")
    print("Example: Tableau Conjugation (In-Place & Out-of-Place)")
    print("---------------------------------------------------")

    # --------------------------------------------------------------------------
    # 1. Setup: Create a Tableau representing a CNOT gate (CX 0->1)
    # --------------------------------------------------------------------------
    var t = Tableau(2)
    t.apply_CX(0, 1)  # Applies CNOT(0, 1)

    # --------------------------------------------------------------------------
    # 2. In-Place Conjugation using `apply_within`
    # --------------------------------------------------------------------------
    # We create a PauliString "XZI". We want to apply the 2-qubit CNOT (qubits 0,1)
    # to the first two qubits of this 3-qubit string.
    var t = Tableau(3)
    t.prepend_H_XZ(0)
    t.prepend_ZCX(1, 2)

    var p = PauliString.from_string("XZI")
    print("Initial Pauli String: ", p)

    var res = t(p)

    # Define target qubits [0, 1]
    var target_qubits = List[Int]()
    target_qubits.append(0)
    target_qubits.append(1)

    print(
        "\n[In-Place] Applying CNOT(0,1) to PauliString 'XZI' on qubits {0, 1}"
    )
    print("  Input:  " + String(p) + " (Phase: " + String(p.global_phase) + ")")

    # Conjugate: P' = T * P * T^-1
    t.apply_within(p, target_qubits)

    # Expected: CNOT * (X0 Z1) * CNOT = - Y0 Y1
    print("  Output: " + String(p) + " (Phase: " + String(p.global_phase) + ")")
    print("  (Expected: YYI, Phase 2 (which is -1))")

    # --------------------------------------------------------------------------
    # 3. Out-of-Place Conjugation using `__call__` / operator()
    # --------------------------------------------------------------------------
    # Create a new PauliString "XZ" matching the tableau size exactly.
    var p_small = PauliString.from_string("XZ")

    print("\n[Out-of-Place] t(p) where t=CNOT(0,1) and p='XZ'")
    print("  Input:  " + String(p_small))

    # Syntax t(p) creates a new PauliString
    var p_out = t(p_small)

    print(
        "  Output: "
        + String(p_out)
        + " (Phase: "
        + String(p_out.global_phase)
        + ")"
    )
    print("  (Expected: YY, Phase 2)")

    print("---------------------------------------------------")
