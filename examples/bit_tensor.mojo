from stimojo.bit_tensor import BitMatrix
from random import randint


fn main():
    # initialize random BitMatrix
    rows = 32
    cols = 64
    var bm = BitMatrix(rows, cols)
    bm.random()
    # print logical and integer representations
    print("logical representation:")
    print(bm.__str__())
    print("column-major integer representation:")
    print(bm._data)
    # transpose
    bm.transpose()
    # logical representation is the same
    print("logical representation:")
    print(bm.__str__())
    # integer representation will be different
    print("row-major integer representation:")
    print(bm._data)
