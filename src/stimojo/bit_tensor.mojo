from memory import alloc, memset, memcpy, UnsafePointer
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from utils import Index
from math import align_up, log2
from sys import simd_width_of
from algorithm import vectorize
from sys import bit_width_of


# compile-time parameters used throughout stimojo modules
comptime int_type = DType.uint64  # can be changed
comptime int_bit_width = bit_width_of[int_type]()  # must be inferred
comptime int_bit_exp = Int(
    log2(SIMD[DType.float64, 1](int_bit_width))
)  # must be inferred
comptime simd_width = simd_width_of[int_type]()  # must be inferred


struct BitVector(
    Copyable, EqualityComparable, ImplicitlyCopyable, Movable, Stringable
):
    """Bit-packed dense vector with data backed by a LayoutTensor."""

    comptime layout_type = Layout.row_major(UNKNOWN_VALUE)

    var n_bits: Int
    var n_words: Int
    var _data: LayoutTensor[int_type, Self.layout_type, MutOrigin.external]

    fn __init__(out self, n_bits: Int):
        # Allocate ptr size aligned up to simd_width
        self.n_bits = n_bits
        self.n_words = (n_bits + int_bit_width - 1) // int_bit_width
        var alloc_size = align_up(self.n_words, simd_width)

        var ptr = alloc[Scalar[int_type]](alloc_size)
        memset(ptr, 0, alloc_size)

        # Define tensor layout at runtime
        var rt_layout = RuntimeLayout[Self.layout_type].row_major(
            Index(alloc_size)
        )
        self._data = LayoutTensor[
            int_type, Self.layout_type, MutOrigin.external
        ](ptr, rt_layout)

    fn __copyinit__(out self, other: BitVector):
        self.n_bits = other.n_bits
        self.n_words = other.n_words

        # Allocate new ptr and runtime layout
        var alloc_size = align_up(self.n_words, simd_width)
        var ptr = alloc[Scalar[int_type]](alloc_size)
        var rt_layout = RuntimeLayout[Self.layout_type].row_major(
            Index(alloc_size)
        )

        # Copy data from other's tensor ptr
        memcpy(dest=ptr, src=other._data.ptr, count=alloc_size)

        # Initialize new tensor
        self._data = LayoutTensor[
            int_type, Self.layout_type, MutOrigin.external
        ](ptr, rt_layout)

    fn __moveinit__(out self, deinit other: BitVector):
        self.n_bits = other.n_bits
        self.n_words = other.n_words
        self._data = other._data

    fn __getitem__(self, idx: Int) -> Bool:
        """Fetches bit state stored at index idx."""
        var word_idx = idx >> int_bit_exp  # word = idx // int_bit_width
        var bit_idx = idx & (int_bit_width - 1)  # bit_idx = idx % int_bit_width
        var word = self._data.ptr.load(word_idx)
        return (
            (word >> bit_idx) & 1
        ) == 1  # shift word to index then AND with mask=1

    fn __setitem__(self, idx: Int, val: Bool):
        """Sets bit state stored at index idx."""
        var word_idx = idx >> int_bit_exp
        var bit_idx = idx & (int_bit_width - 1)
        var mask = (
            Scalar[int_type](1) << bit_idx
        )  # mask of 0s except at bit_idx

        var current_word = self._data.ptr.load(word_idx)
        if val:
            current_word |= mask  # set bit to 1 at bit_idx via OR with mask
        else:
            current_word &= (
                ~mask
            )  # set bit to 0 at bit_idx via AND with NOT mask
        self._data.ptr.store(word_idx, current_word)

    fn __eq__(self, other: BitVector) -> Bool:
        if self.n_bits != other.n_bits:
            return False

        # Compare words
        for i in range(self.n_words):
            if self._data[i] != other._data[i]:
                return False
        return True

    fn __ne__(self, other: BitVector) -> Bool:
        return not (self == other)

    fn __str__(self) -> String:
        """Binary vector represenation."""
        var s = String()
        for i in range(self.n_bits):
            if self[i]:
                s += "1"
            else:
                s += "0"
        return s

    @always_inline
    fn load[width: Int](self, idx: Int) -> SIMD[int_type, width]:
        """SIMD load from LayoutTensor words data."""
        return self._data.load[width](Index(idx))

    @always_inline
    fn store[width: Int](self, idx: Int, val: SIMD[int_type, width]):
        """SIMD store to LayoutTensor words data."""
        self._data.store[width](Index(idx), val)

    fn unsafe_ptr(self) -> UnsafePointer[Scalar[int_type], MutOrigin.external]:
        return self._data.ptr


struct BitMatrix(
    Copyable, EqualityComparable, ImplicitlyCopyable, Movable, Stringable
):
    # Column-Major packed layout
    # Shape is (n_cols, n_words_per_col)
    # n_words_per_col = ceil(n_rows / 64)
    # Effectively transposing the storage to make column operations fast.
    alias layout_type = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var n_rows: Int
    var n_cols: Int
    var n_words_per_col: Int
    var _data: LayoutTensor[int_type, Self.layout_type, MutOrigin.external]

    fn __init__(out self, rows: Int, cols: Int):
        self.n_rows = rows
        self.n_cols = cols
        self.n_words_per_col = (rows + int_bit_width - 1) // int_bit_width
        var alloc_size = cols * self.n_words_per_col

        var ptr = alloc[Scalar[int_type]](alloc_size)
        memset(ptr, 0, alloc_size)

        var rt_layout = RuntimeLayout[Self.layout_type].row_major(
            Index(cols, self.n_words_per_col)
        )
        self._data = LayoutTensor[
            int_type, Self.layout_type, MutOrigin.external
        ](ptr, rt_layout)

    fn __copyinit__(out self, other: BitMatrix):
        self.n_rows = other.n_rows
        self.n_cols = other.n_cols
        self.n_words_per_col = other.n_words_per_col
        var alloc_size = self.n_cols * self.n_words_per_col

        var ptr = alloc[Scalar[int_type]](alloc_size)
        var rt_layout = RuntimeLayout[Self.layout_type].row_major(
            Index(self.n_cols, self.n_words_per_col)
        )

        memcpy(dest=ptr, src=other._data.ptr, count=alloc_size)

        self._data = LayoutTensor[
            int_type, Self.layout_type, MutOrigin.external
        ](ptr, rt_layout)

    fn __moveinit__(out self, deinit other: BitMatrix):
        self.n_rows = other.n_rows
        self.n_cols = other.n_cols
        self.n_words_per_col = other.n_words_per_col
        self._data = other._data

    fn __del__(deinit self):
        if self._data.ptr:
            self._data.ptr.free()

    fn __getitem__(self, r: Int, c: Int) -> Bool:
        # Data is packed along rows (words contain bits for multiple rows of a single column)
        # _data[c, w]
        var w = r >> int_bit_exp
        var b = r & (int_bit_width - 1)
        var word = self._data[c, w][0]
        return ((word >> b) & 1) == 1

    fn __setitem__(mut self, r: Int, c: Int, val: Bool):
        var w = r >> int_bit_exp
        var b = r & (int_bit_width - 1)
        var mask = Scalar[int_type](1) << b
        var word = self._data[c, w][0]
        if val:
            word |= mask
        else:
            word &= ~mask
        self._data[c, w] = word

    fn __eq__(self, other: BitMatrix) -> Bool:
        if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
            return False

        for c in range(self.n_cols):
            for w in range(self.n_words_per_col):
                if self._data[c, w][0] != other._data[c, w][0]:
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
        var total_words = self.n_cols * self.n_words_per_col
        var ptr = self.unsafe_ptr()
        var has_non_zero = False

        @parameter
        fn vec_check[width: Int](i: Int):
            if has_non_zero:
                return
            var v = ptr.load[width=width](i)
            if v.reduce_or() != 0:
                has_non_zero = True

        vectorize[vec_check, simd_width](total_words)
        return not has_non_zero

    fn is_identity(self) -> Bool:
        if self.n_rows != self.n_cols:
            return False

        var all_good = True

        # Check diagonal
        for c in range(self.n_cols):
            if not all_good:
                break

            # We want M[r, r] == 1 for all r.
            # In our layout, M[r, c] is bit r of column c.
            # So for column c, we expect bit c to be 1, and all others 0.

            var diag_w = c >> int_bit_exp
            var diag_b = c & (int_bit_width - 1)
            var expected_diag = Scalar[int_type](1) << diag_b

            for w in range(self.n_words_per_col):
                var val = self._data[c, w][0]
                var expected = expected_diag if w == diag_w else 0
                if val != expected:
                    all_good = False
                    break

        return all_good

    fn unsafe_ptr(self) -> UnsafePointer[Scalar[int_type], MutOrigin.external]:
        return self._data.ptr

    # Access words of a column
    fn col_word(self, c: Int, w: Int) -> Scalar[int_type]:
        var idx = c * self.n_words_per_col + w
        return self._data.ptr.load(idx)

    fn set_col_word(mut self, c: Int, w: Int, val: Scalar[int_type]):
        var idx = c * self.n_words_per_col + w
        self._data.ptr.store(idx, val)

    # Row operations are now slower as they iterate across columns
    fn swap_rows(mut self, r1: Int, r2: Int):
        var w1 = r1 >> int_bit_exp
        var b1 = r1 & (int_bit_width - 1)
        var w2 = r2 >> int_bit_exp
        var b2 = r2 & (int_bit_width - 1)

        var mask1 = Scalar[int_type](1) << b1
        var mask2 = Scalar[int_type](1) << b2

        for c in range(self.n_cols):
            var word1 = self._data[c, w1][0]
            var val1 = (word1 >> b1) & 1

            var word2 = self._data[c, w2][0]
            var val2 = (word2 >> b2) & 1

            if val1 != val2:
                # Toggle bits
                if w1 == w2:
                    # Same word
                    var flip_mask = mask1 | mask2
                    self._data[c, w1] = word1 ^ flip_mask
                else:
                    # Different words
                    self._data[c, w1] = word1 ^ mask1
                    self._data[c, w2] = word2 ^ mask2

    # target_row ^= source_row
    fn xor_row(mut self, target_row: Int, source_row: Int):
        var w_src = source_row >> int_bit_exp
        var b_src = source_row & (int_bit_width - 1)
        var w_tgt = target_row >> int_bit_exp
        var b_tgt = target_row & (int_bit_width - 1)

        var mask_tgt = Scalar[int_type](1) << b_tgt

        for c in range(self.n_cols):
            var val_src = (self._data[c, w_src][0] >> b_src) & 1

            if val_src == 1:
                self._data[c, w_tgt] ^= mask_tgt

    @always_inline
    fn load_col[
        width: Int
    ](self, col: Int, w_idx: Int) -> SIMD[int_type, width]:
        return self._data.load[width](Index(col, w_idx))

    @always_inline
    fn store_col[
        width: Int
    ](self, col: Int, w_idx: Int, val: SIMD[int_type, width]):
        self._data.store[width](Index(col, w_idx), val)
