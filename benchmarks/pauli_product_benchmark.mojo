from stimojo.pauli import PauliString, XZEncoding
import benchmark
from time import time_function


fn main() raises:
    var pauli_terms = Int(5e9)
    var random_encoding_1 = XZEncoding.random_encoding(n_qubits=pauli_terms)
    var random_encoding_2 = XZEncoding.random_encoding(n_qubits=pauli_terms)
    var p1 = PauliString.from_xz_encoding(random_encoding_1)
    var p2 = PauliString.from_xz_encoding(random_encoding_2)

    print(
        "IN-PLACE products of Pauli strings with length={} Billion".format(pauli_terms/1e9)
    )

    @parameter
    fn bench_inplace() raises:
        p1.prod(p2)

    print(
        "Smoke test for 1 iteration:", time_function[bench_inplace]() / 1e9, "s"
    )

    report = benchmark.run[bench_inplace](max_runtime_secs=5)
    report.print(unit=benchmark.Unit.s)

    print(
        "OUT-OF-PLACE products of Pauli strings with length={}".format(
            pauli_terms
        )
    )

    @parameter
    fn bench_outplace() raises:
        _ = p1 * p2

    report = benchmark.run[bench_outplace](max_runtime_secs=5)
    report.print(unit=benchmark.Unit.s)
