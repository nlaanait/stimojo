from stimojo.bit_tensor import BitMatrix
from stimojo import simd_width
import benchmark
from benchmark import keep
from algorithm import vectorize
from python import Python


fn fmt(val: Float64) -> String:
    try:
        var p = Python.import_module("builtins")
        return String(p.format(val, ".2e"))
    except:
        return String(val)


fn main() raises:
    # Header
    print(
        "| Size | transpose (s) | swap_rows (s) | swap_cols (s) | xor_rows (s)"
        " | xor_cols (s) | is_zero (s) | is_identity (s) |"
    )
    print("|---|---|---|---|---|---|---|---|")

    # Sizes: 2**6=64, 2**10=1024, 2**14=16384
    bench_row(64)
    bench_row(1024)
    bench_row(16384)


fn bench_row(N: Int) raises:
    # 0. transpose
    var mat_t = BitMatrix(N, N)
    # Fill diagonal to make it non-trivial
    for k in range(N):
        mat_t[k, k] = True

    @parameter
    fn bench_transpose():
        mat_t.transpose()

    var t_transpose = benchmark.run[bench_transpose](
        min_runtime_secs=0.5, max_runtime_secs=5.0
    ).mean("s")

    # 1. swap_rows
    # Setup: just a matrix
    var mat = BitMatrix(N, N)
    for k in range(N):
        mat[k, k] = True

    # Ensure row-major layout for efficient row ops (BitMatrix init is col-major)
    # If we don't do this, swap_rows will implicitly transpose, dominating the cost.
    # We want to measure the op cost assuming optimal layout.
    mat.transpose()

    @parameter
    fn bench_swap_rows():
        mat.swap_rows(0, 1)

    var t_swap_rows = benchmark.run[bench_swap_rows](
        min_runtime_secs=0.5, max_runtime_secs=5.0
    ).mean("s")

    # 2. xor_rows
    # mat is currently row-major from previous step, optimal for xor_rows
    @parameter
    fn bench_xor_row():
        mat.xor_row(0, 1)

    var t_xor_row = benchmark.run[bench_xor_row](
        min_runtime_secs=0.5, max_runtime_secs=5.0
    ).mean("s")

    # 3. swap_cols
    # Switch back to col-major for optimal column ops
    mat.transpose()

    @parameter
    fn bench_swap_cols():
        mat.swap_cols(0, 1)

    var t_swap_cols = benchmark.run[bench_swap_cols](
        min_runtime_secs=0.5, max_runtime_secs=5.0
    ).mean("s")

    # 4. xor_cols
    # mat is currently col-major from previous step, optimal for xor_cols
    @parameter
    fn bench_xor_col():
        mat.xor_col(0, 1)

    var t_xor_col = benchmark.run[bench_xor_col](
        min_runtime_secs=0.5, max_runtime_secs=5.0
    ).mean("s")

    # 5. is_zero
    # Worst case: matrix is zero (scans all).
    var zero_mat = BitMatrix(N, N)

    @parameter
    fn bench_is_zero():
        keep(zero_mat.is_zero())

    var t_is_zero = benchmark.run[bench_is_zero](
        min_runtime_secs=0.5, max_runtime_secs=5.0
    ).mean("s")

    # 6. is_identity
    # Worst case: matrix IS identity (checks everything).
    var id_mat = BitMatrix(N, N)
    for k in range(N):
        id_mat[k, k] = True

    @parameter
    fn bench_is_id():
        keep(id_mat.is_identity())

    var t_is_id = benchmark.run[bench_is_id](
        min_runtime_secs=0.5, max_runtime_secs=5.0
    ).mean("s")

    print(
        "| "
        + String(N)
        + "x"
        + String(N)
        + " | "
        + fmt(t_transpose)
        + " | "
        + fmt(t_swap_rows)
        + " | "
        + fmt(t_swap_cols)
        + " | "
        + fmt(t_xor_row)
        + " | "
        + fmt(t_xor_col)
        + " | "
        + fmt(t_is_zero)
        + " |"
        + fmt(t_is_id)
        + " |"
    )
