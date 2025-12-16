from memory import alloc, memset, memcpy, UnsafePointer
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from utils import Index
from math import align_up
from sys import simd_width_of
from algorithm import vectorize

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
        var word = self._data[word_idx]
        return ((word >> bit_idx) & 1) == 1

    fn __setitem__(self, idx: Int, val: Bool):
        var word_idx = idx >> bit_exp
        var bit_idx = idx & (bit_width - 1)
        var mask = Scalar[int_type](1) << bit_idx
        
        var current_word = self._data[word_idx]
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
            if self._data[i] != other._data[i]:
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


struct BitMatrix(
    Copyable, EqualityComparable, ImplicitlyCopyable, Movable, Stringable
):
    alias layout_type = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    
    var n_rows: Int
    var n_cols: Int
    var n_words_per_row: Int
    var _data: LayoutTensor[int_type, Self.layout_type, MutOrigin.external]

    fn __init__(out self, rows: Int, cols: Int):
        self.n_rows = rows
        self.n_cols = cols
        self.n_words_per_row = (cols + bit_width - 1) // bit_width
        var alloc_size = rows * self.n_words_per_row
        
        var ptr = alloc[Scalar[int_type]](alloc_size)
        memset(ptr, 0, alloc_size)
        
        var rt_layout = RuntimeLayout[Self.layout_type].row_major(Index(rows, self.n_words_per_row))
        self._data = LayoutTensor[int_type, Self.layout_type, MutOrigin.external](ptr, rt_layout)

    fn __copyinit__(out self, other: BitMatrix):
        self.n_rows = other.n_rows
        self.n_cols = other.n_cols
        self.n_words_per_row = other.n_words_per_row
        var alloc_size = self.n_rows * self.n_words_per_row
        
        var ptr = alloc[Scalar[int_type]](alloc_size)
        var rt_layout = RuntimeLayout[Self.layout_type].row_major(Index(self.n_rows, self.n_words_per_row))
        
        memcpy(dest=ptr, src=other._data.ptr, count=alloc_size)
        
        self._data = LayoutTensor[int_type, Self.layout_type, MutOrigin.external](ptr, rt_layout)

    fn __moveinit__(out self, deinit other: BitMatrix):
        self.n_rows = other.n_rows
        self.n_cols = other.n_cols
        self.n_words_per_row = other.n_words_per_row
        self._data = other._data

    fn __del__(deinit self):
        if self._data.ptr:
            self._data.ptr.free()

    fn __getitem__(self, r: Int, c: Int) -> Bool:
        var w = c >> bit_exp
        var b = c & (bit_width - 1)
        var word = self._data[r, w][0]
        return ((word >> b) & 1) == 1

    fn __setitem__(mut self, r: Int, c: Int, val: Bool):
        var w = c >> bit_exp
        var b = c & (bit_width - 1)
        var mask = Scalar[int_type](1) << b
        var word = self._data[r, w][0]
        if val:
            word |= mask
        else:
            word &= ~mask
        self._data[r, w] = word

    fn __eq__(self, other: BitMatrix) -> Bool:
        if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
            return False
        
        for r in range(self.n_rows):
            for w in range(self.n_words_per_row):
                if self._data[r, w][0] != other._data[r, w][0]:
                    return False
        return True

    fn __ne__(self, other: BitMatrix) -> Bool:
        return not (self == other)

    fn __str__(self) -> String:
        var s = String()
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self[r, c]:
                    s += "1"
                else:
                    s += "0"
            s += "\n"
        return s

    fn is_zero(self) -> Bool:
        var total_words = self.n_rows * self.n_words_per_row
        var ptr = self.unsafe_ptr()
        var has_non_zero = False
        
        @parameter
        fn vec_check[width: Int](i: Int):
            if has_non_zero: return
            var v = ptr.load[width=width](i)
            if v.reduce_or() != 0:
                has_non_zero = True
        
        vectorize[vec_check, simd_width](total_words)
        return not has_non_zero

    fn is_identity(self) -> Bool:
        if self.n_rows != self.n_cols:
            return False
            
        var all_good = True
        
        for r in range(self.n_rows):
            if not all_good: break
            
            var diag_w = r >> bit_exp
            var diag_b = r & (bit_width - 1)
            var expected_diag = Scalar[int_type](1) << diag_b
            
            @parameter
            fn check_row[width: Int](w_start: Int):
                if not all_good: return
                var v = self._data.load[width](Index(r, w_start))
                var expected = SIMD[int_type, width](0)
                
                # Check if diagonal word falls within this vector chunk
                if w_start <= diag_w and diag_w < w_start + width:
                    expected[diag_w - w_start] = expected_diag
                
                if (v ^ expected).reduce_or() != 0:
                    all_good = False

            vectorize[check_row, simd_width](self.n_words_per_row)
            
        return all_good

    fn unsafe_ptr(self) -> UnsafePointer[Scalar[int_type], MutOrigin.external]:
        return self._data.ptr
    
    # 2D access to words
    fn word(self, r: Int, w: Int) -> Scalar[int_type]:
        return self._data[r, w][0]
    
    fn set_word(mut self, r: Int, w: Int, val: Scalar[int_type]):
        self._data[r, w] = val

    # Helper for row swap
    fn swap_rows(mut self, r1: Int, r2: Int):
        @parameter
        fn vec_swap[width: Int](w: Int):
            var v1 = self._data.load[width](Index(r1, w))
            var v2 = self._data.load[width](Index(r2, w))
            self._data.store[width](Index(r1, w), v2)
            self._data.store[width](Index(r2, w), v1)
        
        vectorize[vec_swap, simd_width](self.n_words_per_row)

    # Helper for row xor: target_row ^= source_row
    fn xor_row(mut self, target_row: Int, source_row: Int):
        @parameter
        fn vec_xor[width: Int](w: Int):
            var s = self._data.load[width](Index(source_row, w))
            var t = self._data.load[width](Index(target_row, w))
            self._data.store[width](Index(target_row, w), t ^ s)
        
        vectorize[vec_xor, simd_width](self.n_words_per_row)
    
    @always_inline
    fn load[width: Int](self, row: Int, col: Int) -> SIMD[int_type, width]:
        return self._data.load[width](Index(row, col))

    @always_inline
    fn store[width: Int](self, row: Int, col:Int, val: SIMD[int_type, width]):
        self._data.store[width](Index(row, col), val)