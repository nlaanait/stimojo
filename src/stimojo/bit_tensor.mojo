from memory import alloc, memset, memcpy, UnsafePointer
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from utils import Index
from math import align_up
from sys import simd_width_of

alias int_type = DType.uint64
alias bit_width = 64
alias bit_exp = 6
alias simd_width = simd_width_of[int_type]()

struct BitTensor(
    Copyable, EqualityComparable, ImplicitlyCopyable, Movable, Stringable
):
    alias layout_type = Layout.row_major(UNKNOWN_VALUE)
    
    var num_bits: Int
    var num_words: Int
    var _data: LayoutTensor[int_type, Self.layout_type, MutOrigin.external]

    fn __init__(out self, num_bits: Int):
        self.num_bits = num_bits
        self.num_words = (num_bits + bit_width - 1) // bit_width
        var alloc_size = align_up(self.num_words, simd_width)
        
        var ptr = alloc[Scalar[int_type]](alloc_size)
        memset(ptr, 0, alloc_size)
        
        var rt_layout = RuntimeLayout[Self.layout_type].row_major(Index(alloc_size))
        self._data = LayoutTensor[int_type, Self.layout_type, MutOrigin.external](ptr, rt_layout)

    fn __copyinit__(out self, other: BitTensor):
        self.num_bits = other.num_bits
        self.num_words = other.num_words
        var alloc_size = align_up(self.num_words, simd_width)
        
        var ptr = alloc[Scalar[int_type]](alloc_size)
        var rt_layout = RuntimeLayout[Self.layout_type].row_major(Index(alloc_size))
        
        # Copy data from other
        memcpy(dest=ptr, src=other._data.ptr, count=alloc_size)
        
        self._data = LayoutTensor[int_type, Self.layout_type, MutOrigin.external](ptr, rt_layout)

    fn __moveinit__(out self, deinit other: BitTensor):
        self.num_bits = other.num_bits
        self.num_words = other.num_words
        self._data = other._data

    fn __del__(deinit self):
        if self._data.ptr:
            self._data.ptr.free()

    fn __getitem__(self, idx: Int) -> Bool:
        var word_idx = idx >> bit_exp
        var bit_idx = idx & (bit_width - 1)
        var word = self._data[word_idx][0]
        return ((word >> bit_idx) & 1) == 1

    fn __setitem__(self, idx: Int, val: Bool):
        var word_idx = idx >> bit_exp
        var bit_idx = idx & (bit_width - 1)
        var mask = Scalar[int_type](1) << bit_idx
        
        var current_word = self._data[word_idx][0]
        if val:
            current_word |= mask
        else:
            current_word &= ~mask
        self._data[word_idx] = current_word

    fn __eq__(self, other: BitTensor) -> Bool:
        if self.num_bits != other.num_bits:
            return False
        
        # Compare words
        for i in range(self.num_words):
            if self._data[i][0] != other._data[i][0]:
                return False
        return True

    fn __ne__(self, other: BitTensor) -> Bool:
        return not (self == other)

    fn __str__(self) -> String:
        var s = String()
        for i in range(self.num_bits):
            if self[i]:
                s += "1"
            else:
                s += "0"
        return s

    # Expose SIMD operations for the underlying words (used by PauliString.prod)
    @always_inline
    fn load[width: Int](self, idx: Int) -> SIMD[int_type, width]:
        return self._data.load[width](Index(idx))

    @always_inline
    fn store[width: Int](self, idx: Int, val: SIMD[int_type, width]):
        self._data.store[width](Index(idx), val)

    # Helper to access raw pointer if needed (for bulk copies like in from_xz_encoding)
    fn unsafe_ptr(self) -> UnsafePointer[Scalar[int_type], MutOrigin.external]:
        return self._data.ptr
