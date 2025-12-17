from memory import memset, memcpy, UnsafePointer
from algorithm import vectorize
from utils import Index
from stimojo.pauli import PauliString, Phase, int_type, simd_width
from collections.list import List
from stimojo.bit_tensor import BitVector, BitMatrix
from math import align_up


struct Tableau(Copyable, Movable):
    var n_qubits: Int

    # Storage using BitMatrix
    var _xs_xt: BitMatrix
    var _xs_zt: BitMatrix
    var _xs_signs: BitVector

    var _zs_xt: BitMatrix
    var _zs_zt: BitMatrix
    var _zs_signs: BitVector

    fn __init__(out self, n_qubits: Int):
        self.n_qubits = n_qubits

        self._xs_xt = BitMatrix(n_qubits, n_qubits)
        self._xs_zt = BitMatrix(n_qubits, n_qubits)
        self._xs_signs = BitVector(n_qubits)

        self._zs_xt = BitMatrix(n_qubits, n_qubits)
        self._zs_zt = BitMatrix(n_qubits, n_qubits)
        self._zs_signs = BitVector(n_qubits)

        for k in range(n_qubits):
            self._xs_xt[k, k] = True
            self._zs_zt[k, k] = True

    fn __copyinit__(out self, other: Tableau):
        self.n_qubits = other.n_qubits
        self._xs_xt = other._xs_xt
        self._xs_zt = other._xs_zt
        self._xs_signs = other._xs_signs
        self._zs_xt = other._zs_xt
        self._zs_zt = other._zs_zt
        self._zs_signs = other._zs_signs

    fn __moveinit__(out self, deinit other: Tableau):
        self.n_qubits = other.n_qubits
        self._xs_xt = other._xs_xt^
        self._xs_zt = other._xs_zt^
        self._xs_signs = other._xs_signs^
        self._zs_xt = other._zs_xt^
        self._zs_zt = other._zs_zt^
        self._zs_signs = other._zs_signs^

    fn __str__(self) -> String:
        var s = String()
        s += "xs_xt: \n" + self._xs_xt.__str__()
        s += "xs_zt: \n" + self._xs_zt.__str__()
        s += "zs_xt: \n" + self._zs_xt.__str__()
        s += "zs_zt: \n" + self._zs_zt.__str__()
        s += "xs_signs: " + self._xs_signs.__str__() + "\n"
        s += "zs_signs: " + self._zs_signs.__str__()
        return s

    fn is_pauli_product(self) -> Bool:
        if not self._xs_zt.is_zero():
            return False
        if not self._zs_xt.is_zero():
            return False
        if not self._xs_xt.is_identity():
            return False
        if not self._zs_zt.is_identity():
            return False
        return True

    fn to_pauli_string(self) raises -> PauliString:
        if not self.is_pauli_product():
            raise Error("The Tableau isn't equivalent to a Pauli product.")

        var p = PauliString("I" * self.n_qubits)

        for k in range(self.n_qubits):
            var x_val = self._zs_signs[k]
            var z_val = self._xs_signs[k]

            p.xz_encoding.x[k] = x_val
            p.xz_encoding.z[k] = z_val

        p.pauli_string = String(p.xz_encoding)
        return p^

    fn _swap_rows(mut self, half: Int, r1: Int, r2: Int):
        if half == 0:
            self._xs_xt.swap_rows(r1, r2)
            self._xs_zt.swap_rows(r1, r2)
            var s = self._xs_signs[r1]
            self._xs_signs[r1] = self._xs_signs[r2]
            self._xs_signs[r2] = s
        else:
            self._zs_xt.swap_rows(r1, r2)
            self._zs_zt.swap_rows(r1, r2)
            var s = self._zs_signs[r1]
            self._zs_signs[r1] = self._zs_signs[r2]
            self._zs_signs[r2] = s

    fn x_sign(self, in_qubit: Int) -> Int:
        return Int(self._xs_signs[in_qubit])

    fn set_x_sign(mut self, in_qubit: Int, val: Int):
        self._xs_signs[in_qubit] = val == 1

    fn z_sign(self, in_qubit: Int) -> Int:
        return Int(self._zs_signs[in_qubit])

    fn set_z_sign(mut self, in_qubit: Int, val: Int):
        self._zs_signs[in_qubit] = val == 1

    fn x_out_x(self, in_qubit: Int, out_qubit: Int) -> Int:
        return Int(self._xs_xt[in_qubit, out_qubit])

    fn set_x_out_x(mut self, in_qubit: Int, out_qubit: Int, val: Int):
        self._xs_xt[in_qubit, out_qubit] = val == 1

    fn x_out_z(self, in_qubit: Int, out_qubit: Int) -> Int:
        return Int(self._xs_zt[in_qubit, out_qubit])

    fn set_x_out_z(mut self, in_qubit: Int, out_qubit: Int, val: Int):
        self._xs_zt[in_qubit, out_qubit] = val == 1

    fn z_out_x(self, in_qubit: Int, out_qubit: Int) -> Int:
        return Int(self._zs_xt[in_qubit, out_qubit])

    fn set_z_out_x(mut self, in_qubit: Int, out_qubit: Int, val: Int):
        self._zs_xt[in_qubit, out_qubit] = val == 1

    fn z_out_z(self, in_qubit: Int, out_qubit: Int) -> Int:
        return Int(self._zs_zt[in_qubit, out_qubit])

    fn set_z_out_z(mut self, in_qubit: Int, out_qubit: Int, val: Int):
        self._zs_zt[in_qubit, out_qubit] = val == 1

    # === Operations ===

    fn apply_hadamard(mut self, q: Int):
        for k in range(self.n_qubits):
            # XS
            var x = self._xs_xt[k, q]
            var z = self._xs_zt[k, q]
            if x and z:
                self._xs_signs[k] = not self._xs_signs[k]
            self._xs_xt[k, q] = z
            self._xs_zt[k, q] = x

            # ZS
            x = self._zs_xt[k, q]
            z = self._zs_zt[k, q]
            if x and z:
                self._zs_signs[k] = not self._zs_signs[k]
            self._zs_xt[k, q] = z
            self._zs_zt[k, q] = x

    fn apply_X(mut self, q: Int):
        # Apply X to qubit q:
        # Signs ^= Z_column[q]
        # With new layout, column q is contiguous and packed same as Signs.
        @always_inline
        @parameter
        fn vec_body[width: Int](w_idx: Int):
            # Load column words from ZT matrices
            var v_xs_zt = self._xs_zt.load_col[width=width](q, w_idx)
            var v_zs_zt = self._zs_zt.load_col[width=width](q, w_idx)

            # Load signs words
            var v_xs_signs = self._xs_signs.load[width=width](w_idx)
            var v_zs_signs = self._zs_signs.load[width=width](w_idx)

            # XOR
            self._xs_signs.store[width=width](w_idx, v_xs_signs ^ v_xs_zt)
            self._zs_signs.store[width=width](w_idx, v_zs_signs ^ v_zs_zt)

        vectorize[vec_body, simd_width](self._xs_signs.num_words)

    fn apply_Y(mut self, q: Int):
        # Apply Y to qubit q:
        # Signs ^= X_column[q] ^ Z_column[q]
        @always_inline
        @parameter
        fn vec_body[width: Int](w_idx: Int):
            var v_xs_xt = self._xs_xt.load_col[width=width](q, w_idx)
            var v_xs_zt = self._xs_zt.load_col[width=width](q, w_idx)
            var v_zs_xt = self._zs_xt.load_col[width=width](q, w_idx)
            var v_zs_zt = self._zs_zt.load_col[width=width](q, w_idx)

            var v_xs_signs = self._xs_signs.load[width=width](w_idx)
            var v_zs_signs = self._zs_signs.load[width=width](w_idx)

            self._xs_signs.store[width=width](
                w_idx, v_xs_signs ^ v_xs_xt ^ v_xs_zt
            )
            self._zs_signs.store[width=width](
                w_idx, v_zs_signs ^ v_zs_xt ^ v_zs_zt
            )

        vectorize[vec_body, simd_width](self._xs_signs.num_words)

    fn apply_Z(mut self, q: Int):
        # Apply Z to qubit q:
        # Signs ^= X_column[q]
        @always_inline
        @parameter
        fn vec_body[width: Int](w_idx: Int):
            var v_xs_xt = self._xs_xt.load_col[width=width](q, w_idx)
            var v_zs_xt = self._zs_xt.load_col[width=width](q, w_idx)

            var v_xs_signs = self._xs_signs.load[width=width](w_idx)
            var v_zs_signs = self._zs_signs.load[width=width](w_idx)

            self._xs_signs.store[width=width](w_idx, v_xs_signs ^ v_xs_xt)
            self._zs_signs.store[width=width](w_idx, v_zs_signs ^ v_zs_xt)

        vectorize[vec_body, simd_width](self._xs_signs.num_words)

    fn apply_S(mut self, q: Int):
        # S gate: Z -> Z, X -> Y = iXZ
        # Update signs: Signs ^= X_col & Z_col
        # Update ZT: ZT ^= XT
        @always_inline
        @parameter
        fn vec_body[width: Int](w_idx: Int):
            var v_xs_xt = self._xs_xt.load_col[width=width](q, w_idx)
            var v_xs_zt = self._xs_zt.load_col[width=width](q, w_idx)
            var v_zs_xt = self._zs_xt.load_col[width=width](q, w_idx)
            var v_zs_zt = self._zs_zt.load_col[width=width](q, w_idx)

            # Update signs if X=1 and Z=1
            var flip_xs = v_xs_xt & v_xs_zt
            var flip_zs = v_zs_xt & v_zs_zt

            var v_xs_signs = self._xs_signs.load[width=width](w_idx)
            var v_zs_signs = self._zs_signs.load[width=width](w_idx)
            self._xs_signs.store[width=width](w_idx, v_xs_signs ^ flip_xs)
            self._zs_signs.store[width=width](w_idx, v_zs_signs ^ flip_zs)

            # ZT = ZT ^ XT
            self._xs_zt.store_col[width=width](q, w_idx, v_xs_zt ^ v_xs_xt)
            self._zs_zt.store_col[width=width](q, w_idx, v_zs_zt ^ v_zs_xt)

        vectorize[vec_body, simd_width](self._xs_signs.num_words)

    fn apply_S_dag(mut self, q: Int):
        # S_dag gate: Z -> Z, X -> -Y = -iXZ
        # Update signs: Signs ^= X_col & ~Z_col
        # Update ZT: ZT ^= XT
        @always_inline
        @parameter
        fn vec_body[width: Int](w_idx: Int):
            var v_xs_xt = self._xs_xt.load_col[width=width](q, w_idx)
            var v_xs_zt = self._xs_zt.load_col[width=width](q, w_idx)
            var v_zs_xt = self._zs_xt.load_col[width=width](q, w_idx)
            var v_zs_zt = self._zs_zt.load_col[width=width](q, w_idx)

            var flip_xs = v_xs_xt & (~v_xs_zt)
            var flip_zs = v_zs_xt & (~v_zs_zt)

            var v_xs_signs = self._xs_signs.load[width=width](w_idx)
            var v_zs_signs = self._zs_signs.load[width=width](w_idx)
            self._xs_signs.store[width=width](w_idx, v_xs_signs ^ flip_xs)
            self._zs_signs.store[width=width](w_idx, v_zs_signs ^ flip_zs)

            self._xs_zt.store_col[width=width](q, w_idx, v_xs_zt ^ v_xs_xt)
            self._zs_zt.store_col[width=width](q, w_idx, v_zs_zt ^ v_zs_xt)

        vectorize[vec_body, simd_width](self._xs_signs.num_words)

    fn apply_CX(mut self, c: Int, t: Int):
        # CX: X_c -> X_c X_t, Z_t -> Z_c Z_t
        # XT[t] ^= XT[c]
        # ZT[c] ^= ZT[t]
        # Signs update logic matches the scalar loop
        @always_inline
        @parameter
        fn vec_body[width: Int](w_idx: Int):
            # Load columns
            var xc_xt = self._xs_xt.load_col[width=width](c, w_idx)
            var zc_zt = self._xs_zt.load_col[width=width](c, w_idx)
            var xt_xt = self._xs_xt.load_col[width=width](t, w_idx)
            var zt_zt = self._xs_zt.load_col[width=width](t, w_idx)

            # Calculate sign flips: x1 and z2 and not (x2 != z1)
            # x1=xc_xt, z1=zc_zt, x2=xt_xt, z2=zt_zt
            var flip_xs = xc_xt & zt_zt & (~(xt_xt ^ zc_zt))
            var v_xs_signs = self._xs_signs.load[width=width](w_idx)
            self._xs_signs.store[width=width](w_idx, v_xs_signs ^ flip_xs)

            # Update columns
            self._xs_xt.store_col[width=width](t, w_idx, xt_xt ^ xc_xt)
            self._xs_zt.store_col[width=width](c, w_idx, zc_zt ^ zt_zt)

            # Same for ZS part
            var zc_xt = self._zs_xt.load_col[width=width](c, w_idx)
            var zz_zt = self._zs_zt.load_col[width=width](c, w_idx)
            var zt_xt = self._zs_xt.load_col[width=width](t, w_idx)
            var zt_zt_z = self._zs_zt.load_col[width=width](t, w_idx)

            var flip_zs = zc_xt & zt_zt_z & (~(zt_xt ^ zz_zt))
            var v_zs_signs = self._zs_signs.load[width=width](w_idx)
            self._zs_signs.store[width=width](w_idx, v_zs_signs ^ flip_zs)

            self._zs_xt.store_col[width=width](t, w_idx, zt_xt ^ zc_xt)
            self._zs_zt.store_col[width=width](c, w_idx, zz_zt ^ zt_zt_z)

        vectorize[vec_body, simd_width](self._xs_signs.num_words)

    fn apply_CZ(mut self, c: Int, t: Int):
        @always_inline
        @parameter
        fn vec_body[width: Int](w_idx: Int):
            var xc_xt = self._xs_xt.load_col[width=width](c, w_idx)
            var zc_zt = self._xs_zt.load_col[width=width](c, w_idx)
            var xt_xt = self._xs_xt.load_col[width=width](t, w_idx)
            var zt_zt = self._xs_zt.load_col[width=width](t, w_idx)

            var flip_xs = xc_xt & xt_xt & (zc_zt ^ zt_zt)
            var v_xs_signs = self._xs_signs.load[width=width](w_idx)
            self._xs_signs.store[width=width](w_idx, v_xs_signs ^ flip_xs)

            self._xs_zt.store_col[width=width](c, w_idx, zc_zt ^ xt_xt)
            self._xs_zt.store_col[width=width](t, w_idx, zt_zt ^ xc_xt)

            var zc_xt = self._zs_xt.load_col[width=width](c, w_idx)
            var zz_zt = self._zs_zt.load_col[width=width](c, w_idx)
            var zt_xt = self._zs_xt.load_col[width=width](t, w_idx)
            var zt_zt_z = self._zs_zt.load_col[width=width](t, w_idx)

            var flip_zs = zc_xt & zt_xt & (zz_zt ^ zt_zt_z)
            var v_zs_signs = self._zs_signs.load[width=width](w_idx)
            self._zs_signs.store[width=width](w_idx, v_zs_signs ^ flip_zs)

            self._zs_zt.store_col[width=width](c, w_idx, zz_zt ^ zt_xt)
            self._zs_zt.store_col[width=width](t, w_idx, zt_zt_z ^ zc_xt)

        vectorize[vec_body, simd_width](self._xs_signs.num_words)

    fn apply_CY(mut self, c: Int, t: Int):
        self.apply_S_dag(t)
        self.apply_CX(c, t)
        self.apply_S(t)

    fn prepend_SWAP(mut self, q1: Int, q2: Int):
        # Efficient column swap
        @always_inline
        @parameter
        fn vec_body[width: Int](w_idx: Int):
            # XS
            var v1_xt = self._xs_xt.load_col[width=width](q1, w_idx)
            var v2_xt = self._xs_xt.load_col[width=width](q2, w_idx)
            self._xs_xt.store_col[width=width](q1, w_idx, v2_xt)
            self._xs_xt.store_col[width=width](q2, w_idx, v1_xt)

            var v1_zt = self._xs_zt.load_col[width=width](q1, w_idx)
            var v2_zt = self._xs_zt.load_col[width=width](q2, w_idx)
            self._xs_zt.store_col[width=width](q1, w_idx, v2_zt)
            self._xs_zt.store_col[width=width](q2, w_idx, v1_zt)

            # ZS
            var z1_xt = self._zs_xt.load_col[width=width](q1, w_idx)
            var z2_xt = self._zs_xt.load_col[width=width](q2, w_idx)
            self._zs_xt.store_col[width=width](q1, w_idx, z2_xt)
            self._zs_xt.store_col[width=width](q2, w_idx, z1_xt)

            var z1_zt = self._zs_zt.load_col[width=width](q1, w_idx)
            var z2_zt = self._zs_zt.load_col[width=width](q2, w_idx)
            self._zs_zt.store_col[width=width](q1, w_idx, z2_zt)
            self._zs_zt.store_col[width=width](q2, w_idx, z1_zt)

        vectorize[vec_body, simd_width](self._xs_signs.num_words)

    fn prepend_X(mut self, q: Int):
        self._zs_signs[q] = not self._zs_signs[q]

    fn prepend_Y(mut self, q: Int):
        self._xs_signs[q] = not self._xs_signs[q]
        self._zs_signs[q] = not self._zs_signs[q]

    fn prepend_Z(mut self, q: Int):
        self._xs_signs[q] = not self._xs_signs[q]

    fn _swap_x_z_for_qubit(mut self, q: Int):
        # Column swap between XT and ZT for qubit q
        @always_inline
        @parameter
        fn vec_body[width: Int](w_idx: Int):
            # XS
            var xt = self._xs_xt.load_col[width=width](q, w_idx)
            var zt = self._xs_zt.load_col[width=width](q, w_idx)
            self._xs_xt.store_col[width=width](q, w_idx, zt)
            self._xs_zt.store_col[width=width](q, w_idx, xt)

            # ZS
            var z_xt = self._zs_xt.load_col[width=width](q, w_idx)
            var z_zt = self._zs_zt.load_col[width=width](q, w_idx)
            self._zs_xt.store_col[width=width](q, w_idx, z_zt)
            self._zs_zt.store_col[width=width](q, w_idx, z_xt)

        vectorize[vec_body, simd_width](self._xs_signs.num_words)

        var s = self._xs_signs[q]
        self._xs_signs[q] = self._zs_signs[q]
        self._zs_signs[q] = s

    fn prepend_H_XZ(mut self, q: Int):
        self._swap_x_z_for_qubit(q)

    fn prepend_SQRT_Z_DAG(mut self, q: Int) raises:
        self._mul_rows(0, q, 1, q)

    fn prepend_SQRT_Z(mut self, q: Int) raises:
        self.prepend_SQRT_Z_DAG(q)
        self.prepend_Z(q)

    fn prepend_SQRT_X_DAG(mut self, q: Int) raises:
        self._mul_rows(1, q, 0, q)

    fn prepend_SQRT_X(mut self, q: Int) raises:
        self.prepend_SQRT_X_DAG(q)
        self.prepend_X(q)

    fn prepend_SQRT_Y(mut self, q: Int):
        self._zs_signs[q] = not self._zs_signs[q]
        self._swap_x_z_for_qubit(q)

    fn prepend_SQRT_Y_DAG(mut self, q: Int):
        self._swap_x_z_for_qubit(q)
        self._zs_signs[q] = not self._zs_signs[q]

    fn _xor_pauli_rows(
        mut self,
        target_half: Int,
        target_row: Int,
        source_half: Int,
        source_row: Int,
    ):
        # NOTE: This is slow with column-major storage
        if target_half == 0:
            if source_half == 0:
                self._xs_xt.xor_row(target_row, source_row)
                self._xs_zt.xor_row(target_row, source_row)
            else:
                # Optimized manually
                var w_src = source_row >> 6
                var b_src = source_row & 63
                var w_tgt = target_row >> 6
                var b_tgt = target_row & 63
                var mask_tgt = Scalar[int_type](1) << b_tgt

                for c in range(self.n_qubits):
                    var zs_xt_bit = (
                        self._zs_xt.col_word(c, w_src) >> b_src
                    ) & 1
                    if zs_xt_bit == 1:
                        var val = self._xs_xt.col_word(c, w_tgt)
                        self._xs_xt.set_col_word(c, w_tgt, val ^ mask_tgt)

                    var zs_zt_bit = (
                        self._zs_zt.col_word(c, w_src) >> b_src
                    ) & 1
                    if zs_zt_bit == 1:
                        var val = self._xs_zt.col_word(c, w_tgt)
                        self._xs_zt.set_col_word(c, w_tgt, val ^ mask_tgt)
        else:
            if source_half == 0:
                var w_src = source_row >> 6
                var b_src = source_row & 63
                var w_tgt = target_row >> 6
                var b_tgt = target_row & 63
                var mask_tgt = Scalar[int_type](1) << b_tgt

                for c in range(self.n_qubits):
                    var xs_xt_bit = (
                        self._xs_xt.col_word(c, w_src) >> b_src
                    ) & 1
                    if xs_xt_bit == 1:
                        var val = self._zs_xt.col_word(c, w_tgt)
                        self._zs_xt.set_col_word(c, w_tgt, val ^ mask_tgt)

                    var xs_zt_bit = (
                        self._xs_zt.col_word(c, w_src) >> b_src
                    ) & 1
                    if xs_zt_bit == 1:
                        var val = self._zs_zt.col_word(c, w_tgt)
                        self._zs_zt.set_col_word(c, w_tgt, val ^ mask_tgt)
            else:
                self._zs_xt.xor_row(target_row, source_row)
                self._zs_zt.xor_row(target_row, source_row)

    fn prepend_ZCX(mut self, control: Int, target: Int):
        self._xor_pauli_rows(1, target, 1, control)
        self._xor_pauli_rows(0, control, 0, target)

    fn prepend_H_YZ(mut self, q: Int):
        # ZT[q] ^= XT[q]
        @always_inline
        @parameter
        fn vec_body[width: Int](w_idx: Int):
            var v_z = self._zs_zt.load_col[width=width](q, w_idx)
            var v_x = self._zs_xt.load_col[width=width](q, w_idx)
            self._zs_zt.store_col[width=width](q, w_idx, v_z ^ v_x)

        vectorize[vec_body, simd_width](self._xs_signs.num_words)

        self._zs_signs[q] = not self._zs_signs[q]
        self.prepend_Z(q)

    fn prepend_ZCZ(mut self, control: Int, target: Int):
        self._xor_pauli_rows(0, target, 1, control)
        self._xor_pauli_rows(0, control, 1, target)

    fn prepend_ZCY(mut self, control: Int, target: Int):
        self.prepend_H_YZ(target)
        self.prepend_ZCZ(control, target)
        self.prepend_H_YZ(target)

    fn _row_to_pauli(self, half: Int, row: Int) raises -> PauliString:
        var p = PauliString("I" * self.n_qubits)

        # Reconstructing a PauliString from a scattered row
        var row_w = row >> 6
        var row_b = row & 63
        var words = p.xz_encoding.x.num_words

        if half == 0:
            for w in range(words):
                var packed_x: UInt64 = 0
                var packed_z: UInt64 = 0

                for i in range(64):
                    var q = w * 64 + i
                    if q < self.n_qubits:
                        var x_bit = (
                            self._xs_xt.col_word(q, row_w) >> row_b
                        ) & 1
                        var z_bit = (
                            self._xs_zt.col_word(q, row_w) >> row_b
                        ) & 1

                        packed_x |= x_bit << i
                        packed_z |= z_bit << i

                p.xz_encoding.x._data[w] = packed_x
                p.xz_encoding.z._data[w] = packed_z
        else:
            for w in range(words):
                var packed_x: UInt64 = 0
                var packed_z: UInt64 = 0

                for i in range(64):
                    var q = w * 64 + i
                    if q < self.n_qubits:
                        var x_bit = (
                            self._zs_xt.col_word(q, row_w) >> row_b
                        ) & 1
                        var z_bit = (
                            self._zs_zt.col_word(q, row_w) >> row_b
                        ) & 1

                        packed_x |= x_bit << i
                        packed_z |= z_bit << i

                p.xz_encoding.x._data[w] = packed_x
                p.xz_encoding.z._data[w] = packed_z

        var sign: Bool
        if half == 0:
            sign = self._xs_signs[row]
        else:
            sign = self._zs_signs[row]
        p.global_phase = Phase(2) if sign else Phase(0)
        p.pauli_string = String(p.xz_encoding)
        return p^

    fn _pauli_to_row(mut self, p: PauliString, half: Int, row: Int) raises:
        var row_w = row >> 6
        var row_b = row & 63
        var mask = Scalar[int_type](1) << row_b

        if half == 0:
            for q in range(self.n_qubits):
                var x_val = p.xz_encoding.x[q]
                var z_val = p.xz_encoding.z[q]

                var x_word = self._xs_xt.col_word(q, row_w)
                if x_val:
                    x_word |= mask
                else:
                    x_word &= ~mask
                self._xs_xt.set_col_word(q, row_w, x_word)

                var z_word = self._xs_zt.col_word(q, row_w)
                if z_val:
                    z_word |= mask
                else:
                    z_word &= ~mask
                self._xs_zt.set_col_word(q, row_w, z_word)
        else:
            for q in range(self.n_qubits):
                var x_val = p.xz_encoding.x[q]
                var z_val = p.xz_encoding.z[q]

                var x_word = self._zs_xt.col_word(q, row_w)
                if x_val:
                    x_word |= mask
                else:
                    x_word &= ~mask
                self._zs_xt.set_col_word(q, row_w, x_word)

                var z_word = self._zs_zt.col_word(q, row_w)
                if z_val:
                    z_word |= mask
                else:
                    z_word &= ~mask
                self._zs_zt.set_col_word(q, row_w, z_word)

        if half == 0:
            self._xs_signs[row] = p.global_phase == Phase(2)
        else:
            self._zs_signs[row] = p.global_phase == Phase(2)

    fn _mul_rows(
        mut self,
        target_half: Int,
        target_row: Int,
        source_half: Int,
        source_row: Int,
    ) raises:
        # Optimized _mul_rows could potentially avoid full reconstruction if implemented natively
        var p1 = self._row_to_pauli(target_half, target_row)
        var p2 = self._row_to_pauli(source_half, source_row)
        p1.prod(p2)
        p1.global_phase = Phase(p1.global_phase.log_value & 2)
        self._pauli_to_row(p1, target_half, target_row)

    fn prepend_H_XY(mut self, q: Int) raises:
        # H_XY is tricky, it involves if conditions on X and Z.
        # But for symplectic update:
        # X -> Y, Z -> -Z
        # X -> XZ, Z -> Z (with phase flip if X=1)
        # Wait, H_XY = (X+Z)/sqrt(2)? No, that's H.
        # This implementation was bit-wise.
        # "if x and not z: z=1", etc.
        # This is hard to vectorize across rows if bits depend on each other non-linearly.
        # But we can just iterate.
        for k in range(self.n_qubits):
            var x = self._xs_xt[k, q]
            var z = self._xs_zt[k, q]
            if x and not z:
                self._xs_zt[k, q] = True
            elif not x and z:
                self._xs_signs[k] = not self._xs_signs[k]
            elif x and z:
                self._xs_zt[k, q] = False

            x = self._zs_xt[k, q]
            z = self._zs_zt[k, q]
            if x and not z:
                self._zs_zt[k, q] = True
            elif not x and z:
                self._zs_signs[k] = not self._zs_signs[k]
            elif x and z:
                self._zs_zt[k, q] = False

    fn prepend_C_XYZ(mut self, q: Int) raises:
        self._mul_rows(1, q, 0, q)
        self._swap_x_z_for_qubit(q)

    fn prepend_C_ZYX(mut self, q: Int) raises:
        self._swap_x_z_for_qubit(q)
        self._mul_rows(1, q, 0, q)
        self.prepend_X(q)

    fn prepend_ISWAP(mut self, q1: Int, q2: Int) raises:
        self.prepend_SWAP(q1, q2)
        self.prepend_ZCZ(q1, q2)
        self.prepend_SQRT_Z(q1)
        self.prepend_SQRT_Z(q2)

    fn prepend_ISWAP_DAG(mut self, q1: Int, q2: Int) raises:
        self.prepend_SWAP(q1, q2)
        self.prepend_ZCZ(q1, q2)
        self.prepend_SQRT_Z_DAG(q1)
        self.prepend_SQRT_Z_DAG(q2)

    fn prepend_XCX(mut self, control: Int, target: Int) raises:
        self._mul_rows(1, target, 0, control)
        self._mul_rows(1, control, 0, target)

    fn prepend_XCZ(mut self, control: Int, target: Int):
        self.prepend_ZCX(target, control)

    fn prepend_XCY(mut self, control: Int, target: Int) raises:
        self.prepend_H_XY(target)
        self.prepend_XCX(control, target)
        self.prepend_H_XY(target)

    fn prepend_YCX(mut self, control: Int, target: Int) raises:
        self.prepend_XCY(target, control)

    fn prepend_YCZ(mut self, control: Int, target: Int):
        self.prepend_ZCY(target, control)

    fn prepend_YCY(mut self, control: Int, target: Int):
        self.prepend_H_YZ(control)
        self.prepend_H_YZ(target)
        self.prepend_ZCZ(control, target)
        self.prepend_H_YZ(target)
        self.prepend_H_YZ(control)

    fn prepend_SQRT_ZZ_DAG(mut self, q1: Int, q2: Int) raises:
        self._mul_rows(0, q1, 1, q1)
        self._mul_rows(0, q1, 1, q2)
        self._mul_rows(0, q2, 1, q1)
        self._mul_rows(0, q2, 1, q2)

    fn prepend_SQRT_ZZ(mut self, q1: Int, q2: Int) raises:
        self.prepend_SQRT_ZZ_DAG(q1, q2)
        self.prepend_Z(q1)
        self.prepend_Z(q2)

    fn prepend_SQRT_XX_DAG(mut self, q1: Int, q2: Int) raises:
        self.prepend_H_XZ(q1)
        self.prepend_H_XZ(q2)
        self.prepend_SQRT_ZZ_DAG(q1, q2)
        self.prepend_H_XZ(q2)
        self.prepend_H_XZ(q1)

    fn prepend_SQRT_XX(mut self, q1: Int, q2: Int) raises:
        self.prepend_H_XZ(q1)
        self.prepend_H_XZ(q2)
        self.prepend_SQRT_ZZ(q1, q2)
        self.prepend_H_XZ(q2)
        self.prepend_H_XZ(q1)

    fn prepend_SQRT_YY_DAG(mut self, q1: Int, q2: Int) raises:
        self.prepend_H_YZ(q1)
        self.prepend_H_YZ(q2)
        self.prepend_SQRT_ZZ_DAG(q1, q2)
        self.prepend_H_YZ(q2)
        self.prepend_H_YZ(q1)

    fn prepend_SQRT_YY(mut self, q1: Int, q2: Int) raises:
        self.prepend_H_YZ(q1)
        self.prepend_H_YZ(q2)
        self.prepend_SQRT_ZZ(q1, q2)
        self.prepend_H_YZ(q2)
        self.prepend_H_YZ(q1)

    fn eval_y_obs(self, q: Int) raises -> PauliString:
        var px = self._row_to_pauli(0, q)
        var pz = self._row_to_pauli(1, q)
        var py = px * pz
        py.global_phase += 1
        return py^

    fn eval(self, p: PauliString) raises -> PauliString:
        var res = PauliString("I" * self.n_qubits)
        res.global_phase = p.global_phase.copy()

        for k in range(self.n_qubits):
            var x = p.xz_encoding.x[k]
            var z = p.xz_encoding.z[k]

            if x and not z:
                var row = self._row_to_pauli(0, k)
                res.prod(row)
            elif not x and z:
                var row = self._row_to_pauli(1, k)
                res.prod(row)
            elif x and z:
                var row_x = self._row_to_pauli(0, k)
                var row_z = self._row_to_pauli(1, k)
                res.prod(row_x)
                res.prod(row_z)
                res.global_phase += 1

        return res^

    fn __call__(self, p: PauliString) raises -> PauliString:
        if p.n_qubits != self.n_qubits:
            raise Error(
                "PauliString # of qubits doesn't match the Tableau's # of"
                " qubits!"
            )
        return self.eval(p)

    fn apply_within(
        self, mut target: PauliString, target_qubits: List[Int]
    ) raises:
        if len(target_qubits) != self.n_qubits:
            raise Error("Tableau size must match number of target qubits")

        var inp = PauliString("I" * self.n_qubits)

        for i in range(self.n_qubits):
            var q = target_qubits[i]
            if q >= target.n_qubits:
                raise Error("Target qubit index out of bounds")

            var x = target.xz_encoding.x[q]
            var z = target.xz_encoding.z[q]
            inp.xz_encoding.x[i] = x
            inp.xz_encoding.z[i] = z

        var out = self.eval(inp)

        for i in range(self.n_qubits):
            var q = target_qubits[i]

            var x = out.xz_encoding.x[i]
            var z = out.xz_encoding.z[i]
            target.xz_encoding.x[q] = x
            target.xz_encoding.z[q] = z

        target.global_phase += out.global_phase
        target.pauli_string = String(target.xz_encoding)
