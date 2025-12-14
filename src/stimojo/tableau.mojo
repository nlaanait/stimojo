from memory import memset, memcpy, UnsafePointer
from stimojo.pauli import PauliString, Phase
from collections.list import List

alias int_type = DType.uint8


struct TableauHalf(Copyable, Movable):
    var n_qubits: Int
    var xt: UnsafePointer[UInt8, MutOrigin.external]
    var zt: UnsafePointer[UInt8, MutOrigin.external]
    var signs: UnsafePointer[UInt8, MutOrigin.external]

    fn __init__(
        out self,
        n_qubits: Int,
        xt: UnsafePointer[UInt8, MutOrigin.external],
        zt: UnsafePointer[UInt8, MutOrigin.external],
        signs: UnsafePointer[UInt8, MutOrigin.external],
    ):
        self.n_qubits = n_qubits
        self.xt = xt
        self.zt = zt
        self.signs = signs

    fn swap_rows(self, r1: Int, r2: Int):
        var size = self.n_qubits
        for c in range(size):
            var idx1 = c + r1 * size
            var idx2 = c + r2 * size
            var temp_xt = self.xt[idx1]
            self.xt[idx1] = self.xt[idx2]
            self.xt[idx2] = temp_xt

            var temp_zt = self.zt[idx1]
            self.zt[idx1] = self.zt[idx2]
            self.zt[idx2] = temp_zt

        var temp_sign = self.signs[r1]
        self.signs[r1] = self.signs[r2]
        self.signs[r2] = temp_sign


struct Tableau(Copyable, Movable):
    var n_qubits: Int
    var _xs_xt: UnsafePointer[UInt8, MutOrigin.external]
    var _xs_zt: UnsafePointer[UInt8, MutOrigin.external]
    var _xs_signs: UnsafePointer[UInt8, MutOrigin.external]
    var _zs_xt: UnsafePointer[UInt8, MutOrigin.external]
    var _zs_zt: UnsafePointer[UInt8, MutOrigin.external]
    var _zs_signs: UnsafePointer[UInt8, MutOrigin.external]

    fn __init__(out self, n_qubits: Int):
        self.n_qubits = n_qubits
        var size = n_qubits**2

        self._xs_xt = alloc[UInt8](size)
        self._xs_zt = alloc[UInt8](size)
        self._xs_signs = alloc[UInt8](n_qubits)

        self._zs_xt = alloc[UInt8](size)
        self._zs_zt = alloc[UInt8](size)
        self._zs_signs = alloc[UInt8](n_qubits)

        memset(self._xs_xt, 0, size)
        memset(self._xs_zt, 0, size)
        memset(self._xs_signs, 0, n_qubits)

        memset(self._zs_xt, 0, size)
        memset(self._zs_zt, 0, size)
        memset(self._zs_signs, 0, n_qubits)

        # Set Identity
        for k in range(n_qubits):
            var diag_idx = k + k * n_qubits
            self._xs_xt[diag_idx] = 1
            self._zs_zt[diag_idx] = 1

    fn is_pauli_product(self) -> Bool:
        var size = self.n_qubits * self.n_qubits

        # Check if X part of X output is identity
        # We need to check if _xs_xt is Identity matrix
        for k in range(self.n_qubits):
            for q in range(self.n_qubits):
                var expected = 1 if k == q else 0
                if self._xs_xt[q + k * self.n_qubits] != expected:
                    return False

        # Check if Z part of Z output is identity
        # We need to check if _zs_zt is Identity matrix
        for k in range(self.n_qubits):
            for q in range(self.n_qubits):
                var expected = 1 if k == q else 0
                if self._zs_zt[q + k * self.n_qubits] != expected:
                    return False

        # Check if Z part of X output is zero
        for i in range(size):
            if self._xs_zt[i] != 0:
                return False

        # Check if X part of Z output is zero
        for i in range(size):
            if self._zs_xt[i] != 0:
                return False

        return True

    fn to_pauli_string(self) raises -> PauliString:
        if not self.is_pauli_product():
            raise Error("The Tableau isn't equivalent to a Pauli product.")

        var p = PauliString("I" * self.n_qubits)
        var xs_signs = self.xs().signs
        var zs_signs = self.zs().signs

        for k in range(self.n_qubits):
             # Sign of X stabilizer -> Z component
             # Sign of Z stabilizer -> X component
             var z_val = xs_signs[k]
             var x_val = zs_signs[k]
             p.xz_encoding[k] = (
                 Scalar[DType.uint64](x_val),
                 Scalar[DType.uint64](z_val)
             )

        p.pauli_string = String(p.xz_encoding)
        return p^

    fn xs(self) -> TableauHalf:
        return TableauHalf(
            self.n_qubits, self._xs_xt, self._xs_zt, self._xs_signs
        )

    fn zs(self) -> TableauHalf:
        return TableauHalf(
            self.n_qubits, self._zs_xt, self._zs_zt, self._zs_signs
        )

    fn x_sign(self, in_qubit: Int) -> Int:
        return Int(self._xs_signs[in_qubit])

    fn set_x_sign(self, in_qubit: Int, val: Int):
        self._xs_signs[in_qubit] = val

    fn z_sign(self, in_qubit: Int) -> Int:
        return Int(self._zs_signs[in_qubit])

    fn set_z_sign(self, in_qubit: Int, val: Int):
        self._zs_signs[in_qubit] = val

    fn x_out_x(self, in_qubit: Int, out_qubit: Int) -> Int:
        return Int(self._xs_xt[out_qubit + in_qubit * self.n_qubits])

    fn set_x_out_x(self, in_qubit: Int, out_qubit: Int, val: Int):
        self._xs_xt[out_qubit + in_qubit * self.n_qubits] = val

    fn x_out_z(self, in_qubit: Int, out_qubit: Int) -> Int:
        return Int(self._xs_zt[out_qubit + in_qubit * self.n_qubits])

    fn set_x_out_z(self, in_qubit: Int, out_qubit: Int, val: Int):
        self._xs_zt[out_qubit + in_qubit * self.n_qubits] = val

    fn z_out_x(self, in_qubit: Int, out_qubit: Int) -> Int:
        return Int(self._zs_xt[out_qubit + in_qubit * self.n_qubits])

    fn set_z_out_x(self, in_qubit: Int, out_qubit: Int, val: Int):
        self._zs_xt[out_qubit + in_qubit * self.n_qubits] = val

    fn z_out_z(self, in_qubit: Int, out_qubit: Int) -> Int:
        return Int(self._zs_zt[out_qubit + in_qubit * self.n_qubits])

    fn set_z_out_z(self, in_qubit: Int, out_qubit: Int, val: Int):
        self._zs_zt[out_qubit + in_qubit * self.n_qubits] = val

    # === Operations ===

    fn apply_hadamard(mut self, q: Int):
        for k in range(self.n_qubits):
            var idx = q + k * self.n_qubits

            # Update XS
            var x = self._xs_xt[idx]
            var z = self._xs_zt[idx]
            if x == 1 and z == 1:
                self._xs_signs[k] ^= 1
            self._xs_xt[idx] = z
            self._xs_zt[idx] = x

            # Update ZS
            x = self._zs_xt[idx]
            z = self._zs_zt[idx]
            if x == 1 and z == 1:
                self._zs_signs[k] ^= 1
            self._zs_xt[idx] = z
            self._zs_zt[idx] = x

    fn apply_X(mut self, q: Int):
        for k in range(self.n_qubits):
            var idx = q + k * self.n_qubits
            # Anti-commutes with Z and Y (where Z component is 1)
            if self._xs_zt[idx] == 1:
                self._xs_signs[k] ^= 1
            if self._zs_zt[idx] == 1:
                self._zs_signs[k] ^= 1

    fn apply_Y(mut self, q: Int):
        for k in range(self.n_qubits):
            var idx = q + k * self.n_qubits
            # Anti-commutes with X and Z (where X != Z)
            if self._xs_xt[idx] != self._xs_zt[idx]:
                self._xs_signs[k] ^= 1
            if self._zs_xt[idx] != self._zs_zt[idx]:
                self._zs_signs[k] ^= 1

    fn apply_Z(mut self, q: Int):
        for k in range(self.n_qubits):
            var idx = q + k * self.n_qubits
            # Anti-commutes with X and Y (where X component is 1)
            if self._xs_xt[idx] == 1:
                self._xs_signs[k] ^= 1
            if self._zs_xt[idx] == 1:
                self._zs_signs[k] ^= 1

    fn apply_S(mut self, q: Int):
        for k in range(self.n_qubits):
            var idx = q + k * self.n_qubits
            # Update XS
            var x = self._xs_xt[idx]
            var z = self._xs_zt[idx]
            if x == 1 and z == 1:
                self._xs_signs[k] ^= 1
            self._xs_zt[idx] ^= x

            # Update ZS
            x = self._zs_xt[idx]
            z = self._zs_zt[idx]
            if x == 1 and z == 1:
                self._zs_signs[k] ^= 1
            self._zs_zt[idx] ^= x

    fn apply_S_dag(mut self, q: Int):
        for k in range(self.n_qubits):
            var idx = q + k * self.n_qubits
            # Update XS
            var x = self._xs_xt[idx]
            var z = self._xs_zt[idx]
            if x == 1 and z == 0:
                self._xs_signs[k] ^= 1
            self._xs_zt[idx] ^= x

            # Update ZS
            x = self._zs_xt[idx]
            z = self._zs_zt[idx]
            if x == 1 and z == 0:
                self._zs_signs[k] ^= 1
            self._zs_zt[idx] ^= x

    fn apply_CX(mut self, c: Int, t: Int):
        for k in range(self.n_qubits):
            var c_idx = c + k * self.n_qubits
            var t_idx = t + k * self.n_qubits

            # Update XS
            var x1 = self._xs_xt[c_idx]
            var z1 = self._xs_zt[c_idx]
            var x2 = self._xs_xt[t_idx]
            var z2 = self._xs_zt[t_idx]

            var flip = x1 & z2 & ((x2 ^ z1 ^ 1))
            if flip == 1:
                self._xs_signs[k] ^= 1

            self._xs_xt[t_idx] ^= x1
            self._xs_zt[c_idx] ^= z2

            # Update ZS
            x1 = self._zs_xt[c_idx]
            z1 = self._zs_zt[c_idx]
            x2 = self._zs_xt[t_idx]
            z2 = self._zs_zt[t_idx]

            flip = x1 & z2 & ((x2 ^ z1 ^ 1))
            if flip == 1:
                self._zs_signs[k] ^= 1

            self._zs_xt[t_idx] ^= x1
            self._zs_zt[c_idx] ^= z2

    fn apply_CZ(mut self, c: Int, t: Int):
        for k in range(self.n_qubits):
            var c_idx = c + k * self.n_qubits
            var t_idx = t + k * self.n_qubits

            # Update XS
            var x1 = self._xs_xt[c_idx]
            var z1 = self._xs_zt[c_idx]
            var x2 = self._xs_xt[t_idx]
            var z2 = self._xs_zt[t_idx]

            # sign ^= x1 && x2 && (z1 ^ z2)
            var flip = x1 & x2 & (z1 ^ z2)
            if flip == 1:
                self._xs_signs[k] ^= 1

            self._xs_zt[c_idx] ^= x2
            self._xs_zt[t_idx] ^= x1

            # Update ZS
            x1 = self._zs_xt[c_idx]
            z1 = self._zs_zt[c_idx]
            x2 = self._zs_xt[t_idx]
            z2 = self._zs_zt[t_idx]

            flip = x1 & x2 & (z1 ^ z2)
            if flip == 1:
                self._zs_signs[k] ^= 1

            self._zs_zt[c_idx] ^= x2
            self._zs_zt[t_idx] ^= x1

    fn apply_CY(mut self, c: Int, t: Int):
        # CY = S_dag(t) CX(c, t) S(t)
        self.apply_S_dag(t)
        self.apply_CX(c, t)
        self.apply_S(t)

    fn prepend_SWAP(mut self, q1: Int, q2: Int):
        self.xs().swap_rows(q1, q2)
        self.zs().swap_rows(q1, q2)

    fn prepend_X(mut self, q: Int):
        self.zs().signs[q] ^= 1

    fn prepend_Y(mut self, q: Int):
        self.xs().signs[q] ^= 1
        self.zs().signs[q] ^= 1

    fn prepend_Z(mut self, q: Int):
        self.xs().signs[q] ^= 1

    fn _swap_x_z_for_qubit(mut self, q: Int):
        var n = self.n_qubits
        for k in range(n):  # k is the output_qubit index
            var idx = (
                k + q * n
            )  # This indexes into the row corresponding to input qubit `q`

            var temp_xs_xt = self._xs_xt[idx]
            self._xs_xt[idx] = self._zs_xt[idx]
            self._zs_xt[idx] = temp_xs_xt

            var temp_xs_zt = self._xs_zt[idx]
            self._xs_zt[idx] = self._zs_zt[idx]
            self._zs_zt[idx] = temp_xs_zt

        var temp_xs_sign = self._xs_signs[q]
        self._xs_signs[q] = self._zs_signs[q]
        self._zs_signs[q] = temp_xs_sign

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
        self.zs().signs[q] ^= 1
        self._swap_x_z_for_qubit(q)

    fn prepend_SQRT_Y_DAG(mut self, q: Int):
        self._swap_x_z_for_qubit(q)
        self.zs().signs[q] ^= 1

    fn _xor_pauli_rows(
        mut self,
        half_target_xt: UnsafePointer[UInt8, MutOrigin.external],
        half_target_zt: UnsafePointer[UInt8, MutOrigin.external],
        target_q: Int,
        half_source_xt: UnsafePointer[UInt8, MutOrigin.external],
        half_source_zt: UnsafePointer[UInt8, MutOrigin.external],
        source_q: Int,
    ):
        var n = self.n_qubits
        for k in range(n):
            var idx_target = k + target_q * n
            var idx_source = k + source_q * n

            half_target_xt[idx_target] ^= half_source_xt[idx_source]
            half_target_zt[idx_target] ^= half_source_zt[idx_source]

    fn prepend_ZCX(mut self, control: Int, target: Int):
        self._xor_pauli_rows(
            self._zs_xt, self._zs_zt, target, self._zs_xt, self._zs_zt, control
        )
        self._xor_pauli_rows(
            self._xs_xt, self._xs_zt, control, self._xs_xt, self._xs_zt, target
        )

    fn prepend_H_YZ(mut self, q: Int):
        var n = self.n_qubits
        for k in range(n):  # k is the output_qubit index
            var idx = k + q * n
            self._zs_zt[idx] ^= self._zs_xt[idx]
        self._zs_signs[q] ^= 1
        self.prepend_Z(q)

    fn prepend_ZCZ(mut self, control: Int, target: Int):
        self._xor_pauli_rows(
            self._xs_xt, self._xs_zt, target, self._zs_xt, self._zs_zt, control
        )
        self._xor_pauli_rows(
            self._xs_xt, self._xs_zt, control, self._zs_xt, self._zs_zt, target
        )

    fn prepend_ZCY(mut self, control: Int, target: Int):
        self.prepend_H_YZ(target)
        self.prepend_ZCZ(control, target)
        self.prepend_H_YZ(target)

    fn _row_to_pauli(self, half: Int, row: Int) raises -> PauliString:
        # half: 0 for xs, 1 for zs
        var p = PauliString("I" * self.n_qubits)

        var n = self.n_qubits
        var xt_ptr = self._xs_xt if half == 0 else self._zs_xt
        var zt_ptr = self._xs_zt if half == 0 else self._zs_zt
        var signs_ptr = self._xs_signs if half == 0 else self._zs_signs

        for c in range(n):
            var idx = c + row * n
            p.xz_encoding[c] = (
                Scalar[DType.uint64](xt_ptr[idx]),
                Scalar[DType.uint64](zt_ptr[idx]),
            )

        if signs_ptr[row] == 1:
            p.global_phase = Phase(2)  # -1 phase
        else:
            p.global_phase = Phase(0)

        # Update string representation
        p.pauli_string = String(p.xz_encoding)

        return p^

    fn _pauli_to_row(mut self, p: PauliString, half: Int, row: Int) raises:
        var n = self.n_qubits
        var xt_ptr = self._xs_xt if half == 0 else self._zs_xt
        var zt_ptr = self._xs_zt if half == 0 else self._zs_zt
        var signs_ptr = self._xs_signs if half == 0 else self._zs_signs

        for c in range(n):
            var idx = c + row * n
            var xz = p.xz_encoding[c]
            xt_ptr[idx] = xz[0].cast[DType.uint8]()
            zt_ptr[idx] = xz[1].cast[DType.uint8]()

        if p.global_phase == Phase(2):
            signs_ptr[row] = 1
        else:
            signs_ptr[row] = 0

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
        # Simulate IgnoreAntiCommute: drop imaginary component of phase
        p1.global_phase = Phase(p1.global_phase.log_value & 2)
        self._pauli_to_row(p1, target_half, target_row)

    fn prepend_H_XY(mut self, q: Int) raises:
        var n = self.n_qubits
        for k in range(n):  # Iterate through each generator row
            var xs_idx = q + k * n
            var zs_idx = q + k * n

            # Transform for X-type stabilizers (t.xs[k])
            var x_val_xs = self._xs_xt[xs_idx]
            var z_val_xs = self._xs_zt[xs_idx]
            if x_val_xs == 1 and z_val_xs == 0:  # X -> Y
                self._xs_zt[xs_idx] = 1  # Change X to Y
            elif x_val_xs == 0 and z_val_xs == 1:  # Z -> -Z
                self._xs_signs[k] ^= 1  # Flip sign
            elif x_val_xs == 1 and z_val_xs == 1:  # Y -> X
                self._xs_zt[xs_idx] = 0  # Change Y to X

            # Transform for Z-type stabilizers (t.zs[k])
            var x_val_zs = self._zs_xt[zs_idx]
            var z_val_zs = self._zs_zt[zs_idx]
            if x_val_zs == 1 and z_val_zs == 0:  # X -> Y
                self._zs_zt[zs_idx] = 1  # Change X to Y
            elif x_val_zs == 0 and z_val_zs == 1:  # Z -> -Z
                self._zs_signs[k] ^= 1  # Flip sign
            elif x_val_zs == 1 and z_val_zs == 1:  # Y -> X
                self._zs_zt[zs_idx] = 0  # Change Y to X

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

        # Y = i X Z. px*pz gives XZ. XZ = -i Y.
        # We want T(Y) = T(i X Z) = i T(X) T(Z).
        # py = T(X) * T(Z).
        # result = i * py.
        py.global_phase += 1
        return py^

    fn eval(self, p: PauliString) raises -> PauliString:
        var res = PauliString("I" * self.n_qubits)
        res.global_phase = p.global_phase.copy()

        for k in range(self.n_qubits):
            var xz = p.xz_encoding[k]
            var x = xz[0]
            var z = xz[1]
            if x == 1 and z == 0:
                var row = self._row_to_pauli(0, k)
                res.prod(row)
            elif x == 0 and z == 1:
                var row = self._row_to_pauli(1, k)
                res.prod(row)
            elif x == 1 and z == 1:
                # Y = i X Z
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

        # Construct temporary PauliString initialized to Identity
        var s = String("")
        for _ in range(self.n_qubits):
            s += "I"
        var inp = PauliString(s)

        # Gather
        for i in range(self.n_qubits):
            var q = target_qubits[i]
            if q >= target.n_qubits:
                raise Error("Target qubit index out of bounds")
            inp.xz_encoding[i] = target.xz_encoding[q]

        # Eval
        var out = self.eval(inp)

        # Scatter
        for i in range(self.n_qubits):
            var q = target_qubits[i]
            if q >= target.n_qubits:
                raise Error("Target qubit index out of bounds")
            target.xz_encoding[q] = out.xz_encoding[i]

        # Update Phase
        target.global_phase += out.global_phase
        target.pauli_string = String(target.xz_encoding)

    # === Lifecycle ===

    fn __copyinit__(out self, other: Tableau):
        self.n_qubits = other.n_qubits
        var size = self.n_qubits * self.n_qubits

        self._xs_xt = alloc[UInt8](size)
        self._xs_zt = alloc[UInt8](size)
        self._xs_signs = alloc[UInt8](self.n_qubits)

        self._zs_xt = alloc[UInt8](size)
        self._zs_zt = alloc[UInt8](size)
        self._zs_signs = alloc[UInt8](self.n_qubits)

        memcpy(dest=self._xs_xt, src=other._xs_xt, count=size)
        memcpy(dest=self._xs_zt, src=other._xs_zt, count=size)
        memcpy(dest=self._xs_signs, src=other._xs_signs, count=self.n_qubits)

        memcpy(dest=self._zs_xt, src=other._zs_xt, count=size)
        memcpy(dest=self._zs_zt, src=other._zs_zt, count=size)
        memcpy(dest=self._zs_signs, src=other._zs_signs, count=self.n_qubits)

    fn __moveinit__(out self, deinit other: Tableau):
        self.n_qubits = other.n_qubits
        self._xs_xt = other._xs_xt
        self._xs_zt = other._xs_zt
        self._xs_signs = other._xs_signs
        self._zs_xt = other._zs_xt
        self._zs_zt = other._zs_zt
        self._zs_signs = other._zs_signs

    fn __del__(deinit self):
        if self._xs_xt:
            self._xs_xt.destroy_pointee()
            self._xs_xt.free()
        if self._xs_zt:
            self._xs_zt.destroy_pointee()
            self._xs_zt.free()
        if self._xs_signs:
            self._xs_signs.destroy_pointee()
            self._xs_signs.free()

        if self._zs_xt:
            self._zs_xt.destroy_pointee()
            self._zs_xt.free()
        if self._zs_zt:
            self._zs_zt.destroy_pointee()
            self._zs_zt.free()
        if self._zs_signs:
            self._zs_signs.destroy_pointee()
            self._zs_signs.free()
