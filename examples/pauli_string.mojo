from stimojo.pauli import PauliString


fn main() raises:
    var p1 = PauliString.from_string("ZYX", global_phase=2)
    var p2 = PauliString.from_string("XZY")
    p3 = p1 * p2  # out-of-place product

    print(
        "Product of 2 Pauli strings:\n{} * {} = {}".format(
            String(p1), String(p2), String(p3)
        )
    )

    var exp_phase = p3.global_phase.exponent()

    print("Global phase (log base-i): {}".format(p3.global_phase.value))
    print("Global phase: {} + {}j".format(exp_phase.re, exp_phase.im))
