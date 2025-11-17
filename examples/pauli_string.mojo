from stimojo.pauli import PauliString


fn main() raises:
    var p1 = PauliString("XXX")
    var p2 = PauliString("XXX")

    print("Product of 2 Pauli strings:{} * {}".format(String(p1), String(p2)))

    var p3 = p1 * p2

    print("Result: {}".format(String(p3)))
    print("Global phase: {}".format(p3.global_phase))
