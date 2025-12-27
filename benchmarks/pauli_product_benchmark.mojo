from stimojo.pauli import PauliString, XZEncoding
import benchmark
from time import time_function


fn main() raises:
    var pauli_terms = Int(1e9)
    var p1 = PauliString.random(n_qubits=pauli_terms)
    var p2 = PauliString.random(n_qubits=pauli_terms)
    # print(String(p1.xz_encoding), String(p2.xz_encoding))

    @parameter
    fn bench_inplace() raises:
        p1.prod(p2)

    print(
        "IN-PLACE products of Pauli strings with length={} Billion".format(
            pauli_terms / 1e9
        )
    )

    report = benchmark.run[bench_inplace](max_runtime_secs=15)
    report.print(unit=benchmark.Unit.s)

    print(
        "OUT-OF-PLACE products of Pauli strings with length={} Billion".format(
            pauli_terms / 1e9
        )
    )

    @parameter
    fn bench_outplace() raises:
        _ = p1 * p2

    report = benchmark.run[bench_outplace](max_iters=50)
    report.print(unit=benchmark.Unit.s)
