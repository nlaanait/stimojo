from stimojo.pauli import PauliString
from sys import simd_width_of
from bit import pop_count
from complex import ComplexScalar

alias simd_width = simd_width_of[DType.uint8]()
alias int_type = DType.uint8
alias J = ComplexScalar[DType.int8](
    re=SIMD[DType.int8, 1](0), im=SIMD[DType.int8, 1](1)
)


# TODO: check for commutation
fn commutes(p1: PauliString, p2: PauliString) -> Bool:
    return True


fn commutes(p1: String, p2: String) -> Bool:
    return True


# TODO: partition paulis into commuting groups
fn sort_into_commuting_groups(
    pauli_encodings: List[String],
) -> List[List[String]]:
    ...


fn sort_into_commuting_groups(
    pauli_encodings: List[PauliString],
) -> List[List[PauliString]]:
    ...


fn phase_from_log_base_i(log_base: Int) -> ComplexScalar[DType.int8]:
    var phase = ComplexScalar[DType.int8](
        re=SIMD[DType.int8, 1](1), im=SIMD[DType.int8, 1](0)
    )
    for _ in range(log_base):
        phase *= J

    return phase
