from algorithm import parallelize, vectorize
from collections.list import List
from sys import num_physical_cores, simd_width_of
from bit import pop_count


alias simd_width = simd_width_of[DType.uint8]()
alias int_type = DType.uint8


struct Phase(
    Copyable, EqualityComparable, ImplicitlyCopyable, Movable, Stringable
):
    var log_value: Int

    fn __init__(out self, value: Int) raises:
        self.log_value = value % 4
        self._validate()

    fn __str__(self) -> String:
        var str_phase = String()
        if self.log_value % 4 == 0:
            str_phase = "+"
        elif self.log_value % 4 == 1:
            str_phase = "i"
        elif self.log_value % 4 == 2:
            str_phase = "-"
        elif self.log_value % 4 == 3:
            str_phase = "-i"
        return str_phase

    fn __add__(self, other: Phase) raises -> Phase:
        return Phase(self.log_value + other.log_value)

    fn __add__(self, other: Int) raises -> Phase:
        return self + Phase(other)

    fn __iadd__(mut self, other: Phase):
        self.log_value = (self.log_value + other.log_value) % 4

    fn __iadd__(mut self, other: Int):
        self.log_value = (self.log_value + other) % 4

    fn __eq__(self, other: Phase) -> Bool:
        if self.log_value != other.log_value:
            return False
        return True

    fn __ne__(self, other: Phase) -> Bool:
        return not (self == other)

    fn _validate(self) raises:
        if not (self.log_value in [0, 1, 2, 3]):
            raise Error(
                "Phase should be given in log-i base (mod 4):\n0 -->1, 1-->i, 2"
                " -->-1, 3 -->-i, ..."
            )


struct PauliString(Copyable, EqualityComparable, Movable, Stringable):
    var pauli_string: String
    var x: UnsafePointer[UInt8, MutOrigin.external]
    var z: UnsafePointer[UInt8, MutOrigin.external]
    var n_ops: Int
    var global_phase: Phase

    fn __init__(out self, pauli_string: String, global_phase: Int = 0) raises:
        self.pauli_string = pauli_string.upper()
        self.n_ops = len(self.pauli_string)
        self.x = alloc[UInt8](self.n_ops)
        self.z = alloc[UInt8](self.n_ops)
        self.global_phase = Phase(global_phase)
        self.from_string()

    fn __eq__(self, other: PauliString) -> Bool:
        if self.n_ops != other.n_ops:
            return False
        if self.global_phase != other.global_phase:
            return False

        for i in range(self.n_ops):
            if self.x[i] != other.x[i] or self.z[i] != other.z[i]:
                return False
        return True

    fn __ne__(self, other: PauliString) -> Bool:
        return not (self == other)

    fn from_string(mut self) raises:
        s_up = self.pauli_string.as_bytes()
        for idx in range(self.n_ops):
            s_ = s_up[idx]
            if s_ == ord("I"):
                self.x[idx] = 0
                self.z[idx] = 0
            elif s_ == ord("Z"):
                self.x[idx] = 0
                self.z[idx] = 1
            elif s_ == ord("X"):
                self.x[idx] = 1
                self.z[idx] = 0
            elif s_ == ord("Y"):
                self.x[idx] = 1
                self.z[idx] = 1
            else:
                raise Error(
                    "Encountered Invalid Pauli String!\nValid characters:\n'I':"
                    " Identity.\n'X(x)': Pauli X.\n'Y(y)': Pauli Y.\n'Z(z)':"
                    " Pauli Z.\n Global Phase should be initialized via global_phase arg in base-i log."
                )

    @staticmethod
    fn from_xz_vectors(
        x: UnsafePointer[UInt8, MutOrigin.external],
        z: UnsafePointer[UInt8, MutOrigin.external],
        n_ops: Int,
    ) raises -> PauliString:
        p_str = PauliString.vec_to_string(x, z, n_ops)
        return PauliString(
            p_str,
        )

    @staticmethod
    fn vec_to_string(
        x: UnsafePointer[UInt8],
        z: UnsafePointer[UInt8],
        n_ops: Int,
    ) -> String:
        var result = alloc[UInt8](n_ops)

        @always_inline
        @parameter
        fn compare[simd_width: Int](idx: Int):
            # define SIMD vectors for X,Y,Z,I and all 0's/1's bitstring
            var zeros = SIMD[DType.uint8, simd_width](0)
            var ones = SIMD[DType.uint8, simd_width](1)
            var I_vals = SIMD[DType.uint8, simd_width](ord("I"))
            var X_vals = SIMD[DType.uint8, simd_width](ord("X"))
            var Y_vals = SIMD[DType.uint8, simd_width](ord("Y"))
            var Z_vals = SIMD[DType.uint8, simd_width](ord("Z"))

            # load data from x and z vectors
            var x_chunk = x.load[width=simd_width](idx)
            var z_chunk = z.load[width=simd_width](idx)

            # get boolean mask for each (x,z) tuple to match XYZI
            var I_cond = x_chunk.eq(zeros) & z_chunk.eq(zeros)
            var X_cond = x_chunk.eq(ones) & z_chunk.eq(zeros)
            var Y_cond = x_chunk.eq(ones) & z_chunk.eq(ones)
            var Z_cond = x_chunk.eq(zeros) & z_chunk.eq(ones)

            # use boolean mask to fill otherwise fill with zeros
            var Y_result = Y_cond.select(Y_vals, zeros)
            var I_result = I_cond.select(I_vals, zeros)
            var X_result = X_cond.select(X_vals, zeros)
            var Z_result = Z_cond.select(Z_vals, zeros)

            # add up and store back into result
            var final = Y_result + X_result + Z_result + I_result
            result.store[width=simd_width](idx, final)

        var str = String()

        if n_ops >= simd_width:
            # use vectorized vec to pauli string conversion
            vectorize[compare, simd_width](n_ops)
            for idx in range(n_ops):
                str += chr(Int(result[idx]))
        else:
            # fall back on conditional
            for idx in range(n_ops):
                if x[idx] == 0 and z[idx] == 0:
                    str += "I"
                elif x[idx] == 0 and z[idx] == 1:
                    str += "Z"
                elif x[idx] == 1 and z[idx] == 1:
                    str += "Y"
                elif x[idx] == 1 and z[idx] == 0:
                    str += "X"

        result.destroy_pointee()
        result.free()

        return str

    fn __str__(self) -> String:
        if self.pauli_string == String():
            return PauliString.vec_to_string(self.x, self.z, self.n_ops)
        return String(self.global_phase) + self.pauli_string

    @staticmethod
    fn compute_xor_vector[
        simd_width: Int
    ](
        mut c1: SIMD[int_type, simd_width],
        mut c2: SIMD[int_type, simd_width],
        x: SIMD[int_type, simd_width],
        z: SIMD[int_type, simd_width],
        other_x: SIMD[int_type, simd_width],
        other_z: SIMD[int_type, simd_width],
    ) -> Tuple[
        SIMD[int_type, simd_width],
        SIMD[int_type, simd_width],
    ]:
        var x_result = x ^ other_x
        var z_result = z ^ other_z

        var anti_commutes = (other_x & z) ^ (x & other_z)
        c2 ^= (c1 ^ x_result ^ z_result ^ (x & other_z)) & anti_commutes
        c1 ^= anti_commutes

        return x_result, z_result

    fn __mul__(self, other: PauliString) raises -> PauliString:
        var x_prod = alloc[UInt8](self.n_ops)
        var z_prod = alloc[UInt8](self.n_ops)
        var c1_accum = alloc[UInt8](simd_width)
        var c2_accum = alloc[UInt8](simd_width)

        # Initialize accumulators
        for i in range(simd_width):
            c1_accum[i] = 0
            c2_accum[i] = 0

        @parameter
        fn product[simd_width: Int](idx: Int):
            var c1 = c1_accum.load[width=simd_width](0)
            var c2 = c2_accum.load[width=simd_width](0)

            var x = self.x.load[width=simd_width](idx)
            var z = self.z.load[width=simd_width](idx)

            var other_x = other.x.load[width=simd_width](idx)
            var other_z = other.z.load[width=simd_width](idx)

            x_result, z_result = PauliString.compute_xor_vector(
                c1, c2, x, z, other_x, other_z
            )
            x_prod.store[width=simd_width](idx, x_result)
            z_prod.store[width=simd_width](idx, z_result)
            c1_accum.store[width=simd_width](0, c1)
            c2_accum.store[width=simd_width](0, c2)

        vectorize[product, simd_width](self.n_ops)
        var c1_final = c1_accum.load[width=simd_width](0)
        var c2_final = c2_accum.load[width=simd_width](0)

        phase = (pop_count(c1_final) + 2 * pop_count(c2_final)) % 4

        global_phase = (
            self.global_phase + other.global_phase + Int(phase.reduce_add())
        )

        prod_str = PauliString.vec_to_string(x_prod, z_prod, self.n_ops)

        c1_accum.destroy_pointee()
        c1_accum.free()
        c2_accum.destroy_pointee()
        c2_accum.free()

        return PauliString(prod_str, global_phase=global_phase.log_value)

    fn prod(mut self, other: PauliString) raises:
        var c1_accum = alloc[UInt8](simd_width)
        var c2_accum = alloc[UInt8](simd_width)

        # Initialize accumulators
        for i in range(simd_width):
            c1_accum[i] = 0
            c2_accum[i] = 0

        @parameter
        fn product[simd_width: Int](idx: Int):
            var c1 = c1_accum.load[width=simd_width](0)
            var c2 = c2_accum.load[width=simd_width](0)

            var x = self.x.load[width=simd_width](idx)
            var z = self.z.load[width=simd_width](idx)

            var other_x = other.x.load[width=simd_width](idx)
            var other_z = other.z.load[width=simd_width](idx)

            x_result, z_result = PauliString.compute_xor_vector(
                c1, c2, x, z, other_x, other_z
            )
            self.x.store[width=simd_width](idx, x_result)
            self.z.store[width=simd_width](idx, z_result)
            c1_accum.store[width=simd_width](0, c1)
            c2_accum.store[width=simd_width](0, c2)

        vectorize[product, simd_width](self.n_ops)

        var c1_final = c1_accum.load[width=simd_width](0)
        var c2_final = c2_accum.load[width=simd_width](0)

        phase = (pop_count(c1_final) + 2 * pop_count(c2_final)) % 4

        self.global_phase += other.global_phase + Int(phase.reduce_add())

        self.pauli_string = PauliString.vec_to_string(
            self.x, self.z, self.n_ops
        )

        c1_accum.destroy_pointee()
        c1_accum.free()
        c2_accum.destroy_pointee()
        c2_accum.free()

    fn __del__(deinit self):
        if self.x:
            self.x.destroy_pointee()
            self.x.free()
        if self.z:
            self.z.destroy_pointee()
            self.z.free()
