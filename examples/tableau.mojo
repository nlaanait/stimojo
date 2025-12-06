from stimojo.tableau import Tableau

fn main() raises:
    var n_qubits = 2
    var t = Tableau(n_qubits)
    print(String(t.to_pauli_string()))

    print("Initial Tableau (Identity):")
    print_tableau(t)

    print("\nApplying Hadamard on qubit 1:")
    t.apply_hadamard(1)
    print(String(t.to_pauli_string()))

    print("\nTableau after H on qubit 1:")
    print_tableau(t)

# Helper function to print a Tableau
fn print_tableau(t: Tableau):
    var n_qubits = t.n_qubits
    print("Tableau (", 2 * n_qubits, "x", 2 * n_qubits + 1, "): ")
    
    # Print X outputs
    for k in range(n_qubits):
        var sign = t.x_sign(k)
        if sign == 1:
            print("-", end="")
        else:
            print("+", end="")
            
        for q in range(n_qubits):
            var x = t.x_out_x(k, q)
            var z = t.x_out_z(k, q)
            if x == 1 and z == 1:
                print("Y", end="")
            elif x == 1:
                print("X", end="")
            elif z == 1:
                print("Z", end="")
            else:
                print("I", end="")
        print()

    # Print Z outputs
    for k in range(n_qubits):
        var sign = t.z_sign(k)
        if sign == 1:
            print("-", end="")
        else:
            print("+", end="")
            
        for q in range(n_qubits):
            var x = t.z_out_x(k, q)
            var z = t.z_out_z(k, q)
            if x == 1 and z == 1:
                print("Y", end="")
            elif x == 1:
                print("X", end="")
            elif z == 1:
                print("Z", end="")
            else:
                print("I", end="")
        print()