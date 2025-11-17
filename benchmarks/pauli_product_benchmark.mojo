from stimojo.pauli import PauliString
import benchmark


fn main() raises:
    var repeats = 1e6
    var pauli_string_1 = "IXYZ" * Int(repeats)
    var pauli_string_2 = "YZIX" * Int(repeats)
    var p1 = PauliString(pauli_string_1)
    var p2 = PauliString(pauli_string_2)

    print(
        "IN-PLACE products of Pauli strings with length={}".format(
            len(pauli_string_1)
        )
    )

    @parameter
    fn bench_inplace() raises:
        p1.prod(p2)

    report = benchmark.run[bench_inplace](max_runtime_secs=5)
    report.print(unit=benchmark.Unit.s)

    print(
        "OUT-OF-PLACE products of Pauli strings with length={}".format(
            len(pauli_string_1)
        )
    )

    @parameter
    fn bench_outplace() raises:
        _ = p1 * p2

    report = benchmark.run[bench_outplace](max_runtime_secs=5)
    report.print(unit=benchmark.Unit.s)
