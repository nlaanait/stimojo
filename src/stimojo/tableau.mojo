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
        @always_inline
        @parameter
        fn vec_body[width: Int](qubit: Int):
            var _xs_zt = self._xs_zt.load[width=width](qubit, q)
            var _zs_zt = self._zs_zt.load[width=width](qubit, q)
            var _xs_signs = self._xs_signs.load[width=width](qubit)
            var _zs_signs = self._zs_signs.load[width=width](qubit)
            _xs_signs = _xs_zt.eq(1).select(~_xs_signs, _xs_signs)
            _zs_signs = _zs_zt.eq(1).select(~_zs_signs, _zs_signs)
            self._xs_signs.store[width=width](qubit, _xs_signs)
            self._zs_signs.store[width=width](qubit, _zs_signs)

        vectorize[vec_body, simd_width](self.n_qubits)

    fn apply_Y(mut self, q: Int):
        @always_inline
        @parameter
        fn vec_body[width: Int](qubit: Int):
            var _xs_xt = self._xs_xt.load[width=width](qubit, q)
            var _xs_zt = self._xs_zt.load[width=width](qubit, q)
            var _zs_xt = self._zs_xt.load[width=width](qubit, q)
            var _zs_zt = self._zs_zt.load[width=width](qubit, q)
            var _xs_signs = self._xs_signs.load[width=width](qubit)
            var _zs_signs = self._zs_signs.load[width=width](qubit)
            _xs_signs = _xs_xt.eq(_xs_zt).select(_xs_signs, ~_xs_signs)
            _zs_signs = _zs_xt.eq(_zs_zt).select(_zs_signs, ~_zs_signs)
            self._xs_signs.store[width=width](qubit, _xs_signs)
            self._zs_signs.store[width=width](qubit, _zs_signs)

        vectorize[vec_body, simd_width](self.n_qubits)

    fn apply_Z(mut self, q: Int):
        for k in range(self.n_qubits):
            if self._xs_xt[k, q]:
                self._xs_signs[k] = not self._xs_signs[k]
            if self._zs_xt[k, q]:
                self._zs_signs[k] = not self._zs_signs[k]

    fn apply_S(mut self, q: Int):
        for k in range(self.n_qubits):
            var x = self._xs_xt[k, q]
            var z = self._xs_zt[k, q]
            if x and z:
                self._xs_signs[k] = not self._xs_signs[k]
            self._xs_zt[k, q] = z != x

            x = self._zs_xt[k, q]
            z = self._zs_zt[k, q]
            if x and z:
                self._zs_signs[k] = not self._zs_signs[k]
            self._zs_zt[k, q] = z != x

    fn apply_S_dag(mut self, q: Int):
        for k in range(self.n_qubits):
            var x = self._xs_xt[k, q]
            var z = self._xs_zt[k, q]
            if x and not z:
                self._xs_signs[k] = not self._xs_signs[k]
            self._xs_zt[k, q] = z != x

            x = self._zs_xt[k, q]
            z = self._zs_zt[k, q]
            if x and not z:
                self._zs_signs[k] = not self._zs_signs[k]
            self._zs_zt[k, q] = z != x

    fn apply_CX(mut self, c: Int, t: Int):
        for k in range(self.n_qubits):
            var x1 = self._xs_xt[k, c]
            var z1 = self._xs_zt[k, c]
            var x2 = self._xs_xt[k, t]
            var z2 = self._xs_zt[k, t]

            var flip = x1 and z2 and not (x2 != z1)
            if flip:
                self._xs_signs[k] = not self._xs_signs[k]

            self._xs_xt[k, t] = self._xs_xt[k, t] != x1
            self._xs_zt[k, c] = self._xs_zt[k, c] != z2

            x1 = self._zs_xt[k, c]
            z1 = self._zs_zt[k, c]
            x2 = self._zs_xt[k, t]
            z2 = self._zs_zt[k, t]

            flip = x1 and z2 and not (x2 != z1)
            if flip:
                self._zs_signs[k] = not self._zs_signs[k]

            self._zs_xt[k, t] = self._zs_xt[k, t] != x1
            self._zs_zt[k, c] = self._zs_zt[k, c] != z2

    fn apply_CZ(mut self, c: Int, t: Int):
        for k in range(self.n_qubits):
            var x1 = self._xs_xt[k, c]
            var z1 = self._xs_zt[k, c]
            var x2 = self._xs_xt[k, t]
            var z2 = self._xs_zt[k, t]

            var flip = x1 and x2 and (z1 != z2)
            if flip:
                self._xs_signs[k] = not self._xs_signs[k]

            self._xs_zt[k, c] = self._xs_zt[k, c] != x2
            self._xs_zt[k, t] = self._xs_zt[k, t] != x1

            x1 = self._zs_xt[k, c]
            z1 = self._zs_zt[k, c]
            x2 = self._zs_xt[k, t]
            z2 = self._zs_zt[k, t]

            flip = x1 and x2 and (z1 != z2)
            if flip:
                self._zs_signs[k] = not self._zs_signs[k]

            self._zs_zt[k, c] = self._zs_zt[k, c] != x2
            self._zs_zt[k, t] = self._zs_zt[k, t] != x1

    fn apply_CY(mut self, c: Int, t: Int):
        self.apply_S_dag(t)
        self.apply_CX(c, t)
        self.apply_S(t)

    fn prepend_SWAP(mut self, q1: Int, q2: Int):
        self._swap_rows(0, q1, q2)
        self._swap_rows(1, q1, q2)

    fn prepend_X(mut self, q: Int):
        self._zs_signs[q] = not self._zs_signs[q]

    fn prepend_Y(mut self, q: Int):
        self._xs_signs[q] = not self._xs_signs[q]
        self._zs_signs[q] = not self._zs_signs[q]

    fn prepend_Z(mut self, q: Int):
        self._xs_signs[q] = not self._xs_signs[q]

    fn _swap_x_z_for_qubit(mut self, q: Int):
        var num_words = self._xs_xt.n_words_per_row

        for w in range(num_words):
            var v_xs_xt = self._xs_xt.word(q, w)
            var v_zs_xt = self._zs_xt.word(q, w)
            self._xs_xt.set_word(q, w, v_zs_xt)
            self._zs_xt.set_word(q, w, v_xs_xt)

            var v_xs_zt = self._xs_zt.word(q, w)
            var v_zs_zt = self._zs_zt.word(q, w)
            self._xs_zt.set_word(q, w, v_zs_zt)
            self._zs_zt.set_word(q, w, v_xs_zt)

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
        if target_half == 0:
            if source_half == 0:
                self._xs_xt.xor_row(target_row, source_row)
                self._xs_zt.xor_row(target_row, source_row)
            else:  
                var num_words = self._xs_xt.n_words_per_row
                for w in range(num_words):
                    var s_xt = self._zs_xt.word(source_row, w)
                    var t_xt = self._xs_xt.word(target_row, w)
                    self._xs_xt.set_word(target_row, w, t_xt ^ s_xt)

                    var s_zt = self._zs_zt.word(source_row, w)
                    var t_zt = self._xs_zt.word(target_row, w)
                    self._xs_zt.set_word(target_row, w, t_zt ^ s_zt)
        else:  
            if source_half == 0:
                var num_words = self._xs_xt.n_words_per_row
                for w in range(num_words):
                    var s_xt = self._xs_xt.word(source_row, w)
                    var t_xt = self._zs_xt.word(target_row, w)
                    self._zs_xt.set_word(target_row, w, t_xt ^ s_xt)

                    var s_zt = self._xs_zt.word(source_row, w)
                    var t_zt = self._zs_zt.word(target_row, w)
                    self._zs_zt.set_word(target_row, w, t_zt ^ s_zt)
            else:  
                self._zs_xt.xor_row(target_row, source_row)
                self._zs_zt.xor_row(target_row, source_row)

    fn prepend_ZCX(mut self, control: Int, target: Int):
        self._xor_pauli_rows(1, target, 1, control)
        self._xor_pauli_rows(0, control, 0, target)

    fn prepend_H_YZ(mut self, q: Int):
        var num_words = self._zs_zt.n_words_per_row
        for w in range(num_words):
            var v_z = self._zs_zt.word(q, w)
            var v_x = self._zs_xt.word(q, w)
            self._zs_zt.set_word(q, w, v_z ^ v_x)

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

        var xt_ptr = (
            self._xs_xt.unsafe_ptr() if half == 0 else self._zs_xt.unsafe_ptr()
        )
        var zt_ptr = (
            self._xs_zt.unsafe_ptr() if half == 0 else self._zs_zt.unsafe_ptr()
        )
        var signs_ptr = (
            self._xs_signs.unsafe_ptr() if half
            == 0 else self._zs_signs.unsafe_ptr()
        )

        var words = p.xz_encoding.x.num_words
        var row_stride = self._xs_xt.n_words_per_row

        for w in range(words):
            var idx = row * row_stride + w
            p.xz_encoding.x.store[1](w, xt_ptr[idx])
            p.xz_encoding.z.store[1](w, zt_ptr[idx])

        var bit = (signs_ptr[row >> 6] >> (row & 63)) & 1
        p.global_phase = Phase(2) if bit == 1 else Phase(0)
        p.pauli_string = String(p.xz_encoding)
        return p^

    fn _pauli_to_row(mut self, p: PauliString, half: Int, row: Int) raises:
        var xt_ptr = (
            self._xs_xt.unsafe_ptr() if half == 0 else self._zs_xt.unsafe_ptr()
        )
        var zt_ptr = (
            self._xs_zt.unsafe_ptr() if half == 0 else self._zs_zt.unsafe_ptr()
        )
        var signs_ptr = (
            self._xs_signs.unsafe_ptr() if half
            == 0 else self._zs_signs.unsafe_ptr()
        )

        var words = p.xz_encoding.x.num_words
        var row_stride = self._xs_xt.n_words_per_row

        for w in range(words):
            var idx = row * row_stride + w
            xt_ptr.store(idx, p.xz_encoding.x.load[1](w))
            zt_ptr.store(idx, p.xz_encoding.z.load[1](w))

        var word_idx = row >> 6
        var bit_idx = row & 63
        var mask = Scalar[DType.uint64](1) << bit_idx
        var val = signs_ptr[word_idx]
        if p.global_phase == Phase(2):
            val |= mask
        else:
            val &= ~mask
        signs_ptr.store(word_idx, val)

    fn _mul_rows(
        mut self,
        target_half: Int,
        target_row: Int,
        source_half: Int,
        source_row: Int,
    ) raises:
        var p1 = self._row_to_pauli(target_half, target_row)
        var p2 = self._row_to_pauli(source_half, source_row)
        p1.prod(p2)
        p1.global_phase = Phase(p1.global_phase.log_value & 2)
        self._pauli_to_row(p1, target_half, target_row)

    fn prepend_H_XY(mut self, q: Int) raises:
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
