from stimojo.tableau import Tableau


fn main() raises:
    id_tableau = Tableau()
    # Print the tableau in a 2D matrix format
    # The tableau_dim is 9 (2 * 4 + 1) for n_qubits = 4
    var rows = id_tableau.data.shape[0]()
    var cols = id_tableau.data.shape[1]()

    print("Tableau (", rows, "x", cols, "):")
    for r in range(rows):
        for c in range(cols):
            # Access elements using row and column indices
            print(id_tableau.data[r, c], end=" ")
        print() # New line after each row