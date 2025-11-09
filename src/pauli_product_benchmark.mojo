from stimojo.xzencoding import XZEncoding
import benchmark


fn main() raises:
    var repeats = 1_000_000
    var pauli_string_1 = "IXYZ" * repeats
    var pauli_string_2 = "YZIX" * repeats
    var p1 = XZEncoding(pauli_string_1)
    var p2 = XZEncoding(pauli_string_2)

    print(
        "Benchmark on taking products of Pauli strings with length={}".format(
            len(pauli_string_1)
        )
    )

    @parameter
    fn bench() raises:
        _ = p1 * p2

    report = benchmark.run[bench](max_runtime_secs=5)
    report.print(unit=benchmark.Unit.s)
