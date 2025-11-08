from stimojo.xzencoding import XZEncoding
import benchmark


fn main() raises:
    var repeats = 1_000_000
    var pauli_string_1 = "IXYZ"
    var pauli_string_2 = "YZIX"
    var p1 = XZEncoding(pauli_string_1)
    var p2 = XZEncoding(pauli_string_2)

    var p3 = p1 * p2
    print(String(p3))

    @parameter
    fn bench() raises:
        var p1 = XZEncoding(pauli_string_1 * repeats)
        var p2 = XZEncoding(pauli_string_2 * repeats)
        _ = p1 * p2

    report = benchmark.run[bench](max_runtime_secs=5)
    print(
        "benchmark for XZEncoding:\nComputed products of {} Pauli Strings in"
        .format(repeats * 4),
        report.mean(benchmark.Unit.s),
        "seconds",
    )
