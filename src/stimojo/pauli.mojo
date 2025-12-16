from memory import memset, memcpy, UnsafePointer, alloc
from math import align_up, log2
from algorithm import vectorize
from collections.list import List
from sys import simd_width_of
from bit import pop_count

from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from utils import Index
from .bit_tensor import BitTensor

# compile-time parameters for XZEncoding bit-packing data layout
alias int_type = DType.uint64
alias bit_width = 64  # must match int_type
alias bit_exp = 6  # must be equal to log2(bit_width)
alias simd_width = simd_width_of[int_type]()


struct XZEncoding(
    Copyable, EqualityComparable, ImplicitlyCopyable, Movable, Stringable
):
    var n_qubits: Int
    var n_words: Int
    var x: BitTensor
    var z: BitTensor

    fn __init__(out self, n_qubits: Int):
        self.n_qubits = n_qubits
        self.n_words = (self.n_qubits + bit_width - 1) // bit_width
        self.x = BitTensor(n_qubits)
        self.z = BitTensor(n_qubits)

    fn __str__(self) -> String:
        var s = String()
        for idx in range(self.n_qubits):
            var x_bit = self.x[idx]
            var z_bit = self.z[idx]
            if not x_bit and not z_bit:
                s += "I"
            elif x_bit and not z_bit:
                s += "X"
            elif not x_bit and z_bit:
                s += "Z"
            elif x_bit and z_bit:
                s += "Y"
        return s

    fn __copyinit__(out self, other: XZEncoding):
        self.n_qubits = other.n_qubits
        self.n_words = other.n_words
        self.x = other.x
        self.z = other.z

    fn __moveinit__(out self, deinit other: XZEncoding):
        self.n_qubits = other.n_qubits
        self.n_words = other.n_words
        self.x = other.x^
        self.z = other.z^

    fn __setitem__(
        self, idx: Int, val: Tuple[Bool, Bool]
    ):
        self.x[idx] = val[0]
        self.z[idx] = val[1]
    
    fn __setitem__(
        self, idx: Int, val: Tuple[Scalar[int_type], Scalar[int_type]]
    ):
        self.x[idx] = val[0] == 1
        self.z[idx] = val[1] == 1

    fn __getitem__(self, idx: Int) -> Tuple[Scalar[int_type], Scalar[int_type]]:
        var x_val = Int(self.x[idx])
        var z_val = Int(self.z[idx])
        return (x_val, z_val)

    fn __eq__(self, other: XZEncoding) -> Bool:
        if self.n_qubits != other.n_qubits:
            return False
        return self.x == other.x and self.z == other.z

    fn __ne__(self, other: XZEncoding) -> Bool:
        return not (self == other)

    @always_inline
    fn load[
        simd_width: Int
    ](self, idx: Int) -> Tuple[
        SIMD[int_type, simd_width], SIMD[int_type, simd_width]
    ]:
        return (
            self.x.load[simd_width](idx),
            self.z.load[simd_width](idx),
        )

    @always_inline
    fn store[
        simd_width: Int
    ](
        self,
        idx: Int,
        val: Tuple[SIMD[int_type, simd_width], SIMD[int_type, simd_width]],
    ):
        self.x.store[simd_width](idx, val[0])
        self.z.store[simd_width](idx, val[1])


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
                " -->-1, 3 -->-i"
            )


struct PauliString(
    Copyable, EqualityComparable, ImplicitlyCopyable, Movable, Stringable
):
    var pauli_string: String
    var xz_encoding: XZEncoding
    var n_qubits: Int
    var global_phase: Phase

    fn __init__(out self, pauli_string: String, global_phase: Int = 0) raises:
        self.n_qubits = len(pauli_string)
        self.xz_encoding = XZEncoding(n_qubits=self.n_qubits)
        self.global_phase = Phase(global_phase)
        # store pauli string then xz_encoding
        self.pauli_string = pauli_string.upper()
        self.xz_encode()

    fn __copyinit__(out self, other: PauliString):
        self.pauli_string = other.pauli_string
        self.n_qubits = other.n_qubits
        self.xz_encoding = other.xz_encoding
        self.global_phase = other.global_phase

    fn __moveinit__(out self, deinit other: PauliString):
        self.pauli_string = other.pauli_string
        self.n_qubits = other.n_qubits
        self.xz_encoding = other.xz_encoding^
        self.global_phase = other.global_phase^

    fn __eq__(self, other: PauliString) -> Bool:
        if self.n_qubits != other.n_qubits:
            return False
        if self.global_phase != other.global_phase:
            return False
        return self.xz_encoding == other.xz_encoding

    fn __ne__(self, other: PauliString) -> Bool:
        return not (self == other)

    fn xz_encode(mut self) raises:
        var s_up = self.pauli_string.as_bytes()
        for idx in range(self.n_qubits):
            var char = s_up[idx]
            if char == ord("I"):
                x_val = 0
                z_val = 0
            elif char == ord("Z"):
                x_val = 0
                z_val = 1
            elif char == ord("X"):
                x_val = 1
                z_val = 0
            elif char == ord("Y"):
                x_val = 1
                z_val = 1
            else:
                raise Error(
                    "Encountered Invalid Pauli String!\nValid"
                    " characters:\n'I(i)': Identity.\n'X(x)': Pauli X.\n'Y(y)':"
                    " Pauli Y.\n'Z(z)': Pauli Z.\n Global Phase should be"
                    " initialized via global_phase arg in base-i log."
                )
            self.xz_encoding[idx] = (x_val, z_val)

    @staticmethod
    fn from_xz_encoding(
        input_xz: XZEncoding,
        global_phase: Optional[Int],
    ) raises -> PauliString:
        var p = PauliString(
            "I" * input_xz.n_qubits, global_phase=global_phase.or_else(0)
        )
        p.xz_encoding = input_xz # This will deep copy the BitTensor data
        p.pauli_string = String(p.xz_encoding)
        return p

    fn __str__(self) -> String:
        return String(self.global_phase) + String(self.xz_encoding)

    @staticmethod
    @always_inline
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
        var res = self
        res.prod(other)
        return res

    fn prod(mut self, other: PauliString) raises:
        var accum_ptr = alloc[UInt64](2 * simd_width)
        memset(accum_ptr, 0, 2 * simd_width)

        @parameter
        fn vec_body[width: Int](idx: Int):
            var c1 = accum_ptr.load[width=width](0)
            var c2 = accum_ptr.load[width=width](simd_width)
            var x, z = self.xz_encoding.load[width](idx)
            var ox, oz = other.xz_encoding.load[width](idx)

            var res_x, res_z = PauliString.compute_xor_vector(
                c1, c2, x, z, ox, oz
            )

            self.xz_encoding.store[width](idx, (res_x, res_z))

            accum_ptr.store[width=width](0, c1)
            accum_ptr.store[width=width](simd_width, c2)

        vectorize[vec_body, simd_width](self.xz_encoding.n_words)

        var c1_simd = accum_ptr.load[width=simd_width](0)
        var c2_simd = accum_ptr.load[width=simd_width](simd_width)
        var total_c1 = 0
        var total_c2 = 0
        for i in range(simd_width):
            total_c1 += Int(pop_count(c1_simd[i]))
            total_c2 += Int(pop_count(c2_simd[i]))

        var phase_change = total_c1 + 2 * total_c2
        self.global_phase += other.global_phase + phase_change

        accum_ptr.destroy_pointee()
        accum_ptr.free()
