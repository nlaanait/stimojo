from algorithm import parallelize, vectorize
from collections.list import List
from sys import num_physical_cores, simd_width_of

alias simd_width = simd_width_of[DType.uint8]()
alias int_type = DType.uint8
alias zeros = SIMD[DType.uint8, simd_width](0)
alias ones = SIMD[DType.uint8, simd_width](1)
alias I_vals = SIMD[DType.uint8, simd_width](ord("I"))
alias X_vals = SIMD[DType.uint8, simd_width](ord("X"))
alias Y_vals = SIMD[DType.uint8, simd_width](ord("Y"))
alias Z_vals = SIMD[DType.uint8, simd_width](ord("Z"))


struct XZEncoding(Movable, Stringable):
    var pauli_string: String
    var x: UnsafePointer[Scalar[DType.uint8]]
    var z: UnsafePointer[Scalar[DType.uint8]]
    var n_ops: Int
    var padding: Int

    fn __init__(out self, pauli_string: String) raises:
        self.pauli_string = pauli_string.upper()
        self.n_ops = len(self.pauli_string)
        self.padding = 0
        if self.n_ops < simd_width:
            self.padding = simd_width - self.n_ops
        self.x = UnsafePointer[Scalar[DType.uint8]].alloc(
            self.n_ops + self.padding
        )
        self.z = UnsafePointer[Scalar[DType.uint8]].alloc(
            self.n_ops + self.padding
        )
        self.from_string()

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
                raise Error("Encountered Invalid Pauli String")

    fn vec_to_string(
        self, x: UnsafePointer[UInt8], z: UnsafePointer[UInt8]
    ) -> String:
        var result = UnsafePointer[UInt8].alloc(self.n_ops)

        @parameter
        fn compare[_simd_width: Int](idx: Int):
            var x_chunk = x.load[width=simd_width](idx)
            var z_chunk = z.load[width=simd_width](idx)
            var I_cond = x_chunk.eq(zeros) & z_chunk.eq(zeros)
            var X_cond = x_chunk.eq(ones) & z_chunk.eq(zeros)
            var Y_cond = x_chunk.eq(ones) & z_chunk.eq(ones)
            var Z_cond = x_chunk.eq(zeros) & z_chunk.eq(ones)
            var Y_result = Y_cond.select(Y_vals, zeros)
            var I_result = I_cond.select(I_vals, zeros)
            var X_result = X_cond.select(X_vals, zeros)
            var Z_result = Z_cond.select(Z_vals, zeros)
            var final = Y_result + X_result + Z_result + I_result
            result.store[width=simd_width](idx, final)

        vectorize[compare, simd_width](self.n_ops)

        var str = String()
        for idx in range(self.n_ops):
            str += chr(Int(result[idx]))

        result.destroy_pointee()
        result.free()

        return str[:self.n_ops]

    fn __str__(self) -> String:
        if self.pauli_string == String():
            return self.vec_to_string(self.x, self.z)
        return self.pauli_string

    fn __mul__(self, other: XZEncoding) raises -> XZEncoding:
        var x_prod = self.x.copy()
        var z_prod = self.z.copy()

        @parameter
        fn compute_xor_vector[simd_width: Int](idx: Int):
            var x_result = x_prod[idx] ^ other.x[idx]
            var z_result = z_prod[idx] ^ other.z[idx]
            x_prod.store[width=simd_width](idx, x_result)
            z_prod.store[width=simd_width](idx, z_result)

        vectorize[compute_xor_vector, simd_width](self.n_ops)
        prod_str = self.vec_to_string(x_prod, z_prod)
        return XZEncoding(prod_str)

    fn __del__(deinit self):
        if self.x:
            self.x.destroy_pointee()
            self.x.free()
        if self.z:
            self.z.destroy_pointee()
            self.z.free()
