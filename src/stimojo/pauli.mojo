from memory import memset, memcpy, UnsafePointer, alloc
from math import align_up, log2
from algorithm import vectorize
from collections.list import List
from sys import simd_width_of
from bit import pop_count
from random import randint, seed
from complex import ComplexScalar

from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from utils import Index
from .bit_tensor import BitVector, int_type, simd_width, int_bit_width


struct XZEncoding(
    Copyable, EqualityComparable, ImplicitlyCopyable, Movable, Stringable
):
    """Stores the X and Z components of a Pauli string using a bit-packed representation.

    The encoding uses two bit vectors of length `n_qubits`:
    - `x` bits: 1 if the Pauli operator has an X component (X or Y).
    - `z` bits: 1 if the Pauli operator has a Z component (Z or Y).

    Mapping:
    - I: x=0, z=0
    - X: x=1, z=0
    - Z: x=0, z=1
    - Y: x=1, z=1

    Example:
    ```mojo
    from stimojo.pauli import XZEncoding

    var encoding = XZEncoding(2)
    # Set qubit 0 to X (x=1, z=0)
    encoding[0] = (1, 0)
    # Set qubit 1 to Z (x=0, z=1)
    encoding[1] = (0, 1)
    print(String(encoding)) # Prints "XZ"
    ```
    """

    var n_qubits: Int
    var x: BitVector
    var z: BitVector

    fn __init__(out self, n_qubits: Int):
        """Initializes an empty XZEncoding (all Identity).

        Args:
            n_qubits: The number of qubits.
        """
        self.n_qubits = n_qubits
        self.x = BitVector(n_qubits)
        self.z = BitVector(n_qubits)

    fn __str__(self) -> String:
        """Pauli String Representation."""
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

    @staticmethod
    fn random_encoding(n_qubits: Int) -> XZEncoding:
        seed()
        encoding = XZEncoding(n_qubits)
        randint[int_type](
            encoding.x.unsafe_ptr(),
            encoding.x.n_words,
            int_bit_width,
            2**int_bit_width,
        )
        randint[int_type](
            encoding.z.unsafe_ptr(),
            encoding.x.n_words,
            int_bit_width,
            2**int_bit_width,
        )
        return encoding

    fn __copyinit__(out self, other: XZEncoding):
        self.n_qubits = other.n_qubits
        self.x = other.x
        self.z = other.z

    fn __moveinit__(out self, deinit other: XZEncoding):
        self.n_qubits = other.n_qubits
        self.x = other.x^
        self.z = other.z^

    fn __setitem__(self, idx: Int, val: Tuple[Bool, Bool]):
        self.x[idx] = val[0]
        self.z[idx] = val[1]

    fn __setitem__(self, idx: Int, val: Tuple[Int, Int]):
        self.x[idx] = val[0] == 1
        self.z[idx] = val[1] == 1

    fn __setitem__(
        self, idx: Int, val: Tuple[Scalar[int_type], Scalar[int_type]]
    ):
        self.x[idx] = val[0] == 1
        self.z[idx] = val[1] == 1

    fn __getitem__(self, idx: Int) -> Tuple[Scalar[int_type], Scalar[int_type]]:
        var x_bool = self.x[idx]
        var z_bool = self.z[idx]
        var x_val = Scalar[int_type](x_bool)
        var z_val = Scalar[int_type](z_bool)
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
    """Represents a phase as a log of base i.

    The phase is stored as an integer `k` representing `i^k`.
    - 0 -> 1
    - 1 -> i
    - 2 -> -1
    - 3 -> -i

    Example:
    ```mojo
    from stimojo.pauli import Phase

    var p = Phase(1) # represents i
    print(String(p)) # Prints "i"
    var p2 = p + 2 # i * (-1) = -i -> log value 3
    print(String(p2)) # Prints "-i"
    ```
    """

    comptime J = ComplexScalar[DType.int8](
        re=SIMD[DType.int8, 1](0), im=SIMD[DType.int8, 1](1)
    )
    var value: Int

    fn __init__(out self, value: Int) raises:
        """Initializes the Phase.

        Args:
            value: The exponent of i (will be taken modulo 4).
        """
        self.value = value % 4
        self._validate()

    fn __str__(self) -> String:
        var str_phase = String()
        if self.value % 4 == 0:
            str_phase = "+"
        elif self.value % 4 == 1:
            str_phase = "i"
        elif self.value % 4 == 2:
            str_phase = "-"
        elif self.value % 4 == 3:
            str_phase = "-i"
        return str_phase

    fn __add__(self, other: Phase) raises -> Phase:
        return Phase(self.value + other.value)

    fn __add__(self, other: Int) raises -> Phase:
        return self + Phase(other)

    fn __iadd__(mut self, other: Phase):
        self.value = (self.value + other.value) % 4

    fn __iadd__(mut self, other: Int):
        self.value = (self.value + other) % 4

    fn __eq__(self, other: Phase) -> Bool:
        if self.value != other.value:
            return False
        return True

    fn __ne__(self, other: Phase) -> Bool:
        return not (self == other)

    fn _validate(self) raises:
        if not (self.value in [0, 1, 2, 3]):
            raise Error(
                "Phase should be given in log-i base (mod 4):\n0 -->1, 1-->i, 2"
                " -->-1, 3 -->-i"
            )

    fn exponent(self) -> ComplexScalar[DType.int8]:
        var exponent = ComplexScalar[DType.int8](
            re=SIMD[DType.int8, 1](1), im=SIMD[DType.int8, 1](0)
        )
        for _ in range(self.value):
            exponent *= self.J
        return exponent


struct PauliString(
    Copyable, EqualityComparable, ImplicitlyCopyable, Movable, Stringable
):
    """Represents a tensor products of Paulis with a global phase.

    A PauliString consists of an `XZEncoding` for the operator on each qubit
    and a global `Phase`.

    Example:
    ```mojo
    from stimojo.pauli import PauliString

    # Create from string "XY" with phase i (value=1)
    var p1 = PauliString.from_string("XY", 1)
    print(String(p1)) # Prints "iXY"
    ```
    """

    var pauli_string: String
    var xz_encoding: XZEncoding
    var n_qubits: Int
    var global_phase: Phase

    fn __init__(
        out self,
        n_qubits: Int,
        global_phase: Int = 0,
    ) raises:
        """Initializes a PauliString with identity operators.

        Args:
            n_qubits: The number of qubits.
            global_phase: The global phase exponent k (for i^k). Defaults to 0 (+1).
        """
        self.n_qubits = n_qubits
        self.xz_encoding = XZEncoding(self.n_qubits)
        self.global_phase = Phase(global_phase)
        self.pauli_string = ""  # don't populate until __str__() is invoked

    @staticmethod
    fn random(n_qubits: Int) raises -> PauliString:
        """Generates a random PauliString.

        Args:
            n_qubits: The number of qubits.
        """
        var p = PauliString(n_qubits)
        p.xz_encoding = XZEncoding.random_encoding(n_qubits)
        return p

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

    @staticmethod
    fn from_string(
        pauli_string: String,
        global_phase: Optional[Int] = None,
    ) raises -> PauliString:
        """Creates a PauliString from a string representation.

        Args:
            pauli_string: String containing characters 'I', 'X', 'Y', 'Z'.
            global_phase: Optional initial phase exponent. Defaults to 0.

        Returns:
            The constructed PauliString.

        Raises:
            Error: If an invalid character is encountered.
        """
        # Initialize PauliString
        p = PauliString(
            n_qubits=len(pauli_string), global_phase=global_phase.or_else(0)
        )

        # Store pauli string then xz_encoded it
        p.pauli_string = pauli_string.upper()
        var s_up = p.pauli_string.as_bytes()
        for idx in range(p.n_qubits):
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
            p.xz_encoding[idx] = (x_val, z_val)

        return p

    @staticmethod
    fn from_xz_encoding(
        input_xz: XZEncoding,
        global_phase: Optional[Int] = None,
    ) raises -> PauliString:
        """Creates a PauliString from an existing XZEncoding."""
        var p = PauliString(
            input_xz.n_qubits,
            global_phase=global_phase.or_else(0),
        )
        p.xz_encoding = input_xz  # This will deep copy the BitVector data
        return p

    fn __str__(self) -> String:
        """Pauli String representation with phase prefix: (0:+,1:i,2:-,3:-i)."""
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
        """Computes XOR of 2 XZEncodings across 1 SIMD lane while tracking phase.
        """
        var x_result = x ^ other_x
        var z_result = z ^ other_z

        var anti_commutes = (other_x & z) ^ (x & other_z)
        c2 ^= (c1 ^ x_result ^ z_result ^ (x & other_z)) & anti_commutes
        c1 ^= anti_commutes

        return x_result, z_result

    fn __mul__(self, other: PauliString) raises -> PauliString:
        """Out-of-Place product of 2 PauliStrings."""
        var res = self  # deep copy
        res.prod(other)  # invokes in-place prod on copy
        return res

    fn prod(mut self, other: PauliString) raises:
        """In-Place product of 2 PauliStrings (self = self * other).

        This method updates the current PauliString by multiplying it with another.
        It computes the new XZ components using bitwise operations and updates
        the global phase based on commutation relations.

        Args:
            other: The PauliString to multiply with.
        """

        # allocate ptr with 2 * simd_width to store accumulated phase factors
        var accum_ptr = alloc[Scalar[int_type]](2 * simd_width)
        memset(accum_ptr, 0, 2 * simd_width)

        # function to compute Pauli products across a simd lane
        @parameter
        fn prod[width: Int](idx: Int):
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

        # map prod across n_words of BitVector
        vectorize[prod, simd_width](self.xz_encoding.x.n_words)

        # reduce to find the accumulated phase
        var c1_simd = accum_ptr.load[width=simd_width](0)
        var c2_simd = accum_ptr.load[width=simd_width](simd_width)
        var total_c1 = 0
        var total_c2 = 0
        for i in range(simd_width):
            total_c1 += Int(pop_count(c1_simd[i]))
            total_c2 += Int(pop_count(c2_simd[i]))

        var phase_change = total_c1 + 2 * total_c2

        # update the phase (recall: log base-i)
        self.global_phase += other.global_phase + phase_change

        # free up accumulated phase ptr
        accum_ptr.destroy_pointee()
        accum_ptr.free()
