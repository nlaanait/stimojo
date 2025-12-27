from sys.param_env import env_get_dtype
from sys import bit_width_of, simd_width_of
from math import log2

# compile-time parameters used throughout stimojo modules

# defined during build_time
comptime int_type = env_get_dtype["STIMOJO_INT_TYPE", DType.uint64]()
# inferred from int_type
comptime int_bit_width = bit_width_of[int_type]()
comptime int_bit_exp = Int(log2(SIMD[DType.float64, 1](int_bit_width)))
comptime simd_width = simd_width_of[int_type]()
