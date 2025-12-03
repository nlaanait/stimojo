from layout import Layout, LayoutTensor
from memory import memset

alias n_qubits = 4
alias int_type = DType.uint8
alias tableau_row_dim = 2 * n_qubits
alias tableau_col_dim = 2 * n_qubits + 1
alias tableau_layout = Layout.col_major(tableau_row_dim, tableau_col_dim)


struct Tableau(Copyable, Movable):
    var n_qubits: Int
    var data: LayoutTensor[int_type, tableau_layout, MutOrigin.external]
    var buff_size: Int

    def __init__(out self):
        self.n_qubits = n_qubits
        self.buff_size = tableau_col_dim * tableau_row_dim
        var ptr = alloc[UInt8](self.buff_size)
        memset(ptr, 0, self.buff_size)
        self.data = LayoutTensor[int_type, tableau_layout, MutOrigin.external](
            ptr
        )
        # Initialize to all-zero state (identity)
        for r in range(tableau_row_dim):
            for c in range(tableau_col_dim):
                if c == r:
                    self.data[r, c] = 1
