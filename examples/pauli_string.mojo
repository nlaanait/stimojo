from stimojo.pauli import PauliString
from stimojo.ops import phase_from_log_base_i
from math import log2
from complex import ComplexScalar
from python import Python


fn main() raises:
    var p1 = PauliString(3, "ZYX")
    var p2 = PauliString(3, "XZY")

    print("Product of 2 Pauli strings:{} * {}".format(String(p1), String(p2)))

    p3 = p1 * p2

    var exp_phase = phase_from_log_base_i(p3.global_phase.log_value)

    print("Result: {}".format(String(p3)))
    print("Global phase (log base-i): {}".format(p3.global_phase.log_value))
    print("Global phase: {} + {}j".format(exp_phase.re, exp_phase.im))
