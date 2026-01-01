from memory import alloc, memset, memcpy, UnsafePointer
from layout import Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from utils import Index
from math import align_up, log2
from sys import simd_width_of, bit_width_of
from algorithm import vectorize
from sys.param_env import env_get_dtype

from . import int_type, int_bit_exp, int_bit_width, simd_width


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
            (word >> bit_idx) & Scalar[int_type](1)
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
    # Default is Column-Major packed layout
    # Shape is (n_cols, n_words_per_col)
    # n_words_per_col = ceil(n_rows / 64)
    comptime layout_type = Layout.col_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var n_rows: Int
    var n_cols: Int
    var n_words_per_col: Int
    var _data: LayoutTensor[int_type, Self.layout_type, MutOrigin.external]
    var qubit_layout: Bool

    fn __init__(out self, rows: Int, cols: Int):
        self.n_rows = rows
        self.n_cols = cols
        self.n_words_per_col = (rows + int_bit_width - 1) // int_bit_width
        var alloc_size = cols * self.n_words_per_col

        var ptr = alloc[Scalar[int_type]](alloc_size)
        memset(ptr, 0, alloc_size)

        var rt_layout = RuntimeLayout[Self.layout_type].col_major(
            Index(cols, self.n_words_per_col)
        )
        self._data = LayoutTensor[
            int_type, Self.layout_type, MutOrigin.external
        ](ptr, rt_layout)
        self.qubit_layout = True

    fn __copyinit__(out self, other: BitMatrix):
        # TODO: handle case of src/target with different memory layouts.
        self.n_rows = other.n_rows
        self.n_cols = other.n_cols
        self.n_words_per_col = other.n_words_per_col
        self.qubit_layout = other.qubit_layout
        var alloc_size = self.n_cols * self.n_words_per_col

        var ptr = alloc[Scalar[int_type]](alloc_size)
        var rt_layout = RuntimeLayout[Self.layout_type].col_major(
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
        self.qubit_layout = other.qubit_layout

    fn __del__(deinit self):
        if self._data.ptr:
            self._data.ptr.free()

    fn __getitem__(self, r: Int, c: Int) -> Bool:
        # Data is packed along rows (words contain bits for multiple rows of a single column)
        # _data[c, w]
        if self.qubit_layout:
            var w = r >> int_bit_exp
            var b = r & (int_bit_width - 1)
            var word = self._data[c, w][0]
            return ((word >> b) & Scalar[int_type](1)) == 1

        # Data is packed along cols (words contains bits for multiple cols of a single row)
        # _data[r,w]
        else:
            var w = c >> int_bit_exp
            var b = c & (int_bit_width - 1)
            var word = self._data[r, w][0]
            return ((word >> b) & Scalar[int_type](1)) == 1

    fn __setitem__(mut self, r: Int, c: Int, val: Bool):
        var major = c if self.qubit_layout else r
        var minor = r if self.qubit_layout else c

        var w = minor >> int_bit_exp
        var b = minor & (int_bit_width - 1)
        var mask = Scalar[int_type](1) << b

        var word = self._data[major, w][0]
        if val:
            word |= mask
        else:
            word &= ~mask
        self._data[major, w] = word

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

    fn transpose(mut self):
        var current_major = self.n_cols if self.qubit_layout else self.n_rows
        var current_minor = self.n_rows if self.qubit_layout else self.n_cols

        var new_major = current_minor
        var new_minor = current_major
        var new_words_per_major = (
            new_minor + int_bit_width - 1
        ) // int_bit_width
        var new_alloc_size = new_major * new_words_per_major

        var new_ptr = alloc[Scalar[int_type]](new_alloc_size)
        memset(new_ptr, 0, new_alloc_size)

        for c in range(current_major):
            for w in range(self.n_words_per_col):
                var word = self._data[c, w][0]
                # Iterate bits in the word
                for b in range(int_bit_width):
                    var r = (w << int_bit_exp) + b
                    if r < current_minor:
                        if ((word >> b) & Scalar[int_type](1)) == 1:
                            # Set bit at (r, c) in new matrix
                            var dest_w = c >> int_bit_exp
                            var dest_b = c & (int_bit_width - 1)
                            var dest_idx = r * new_words_per_major + dest_w
                            var val = new_ptr.load(dest_idx)
                            new_ptr.store(
                                dest_idx, val | (Scalar[int_type](1) << dest_b)
                            )

        if self._data.ptr:
            self._data.ptr.destroy_pointee()
            self._data.ptr.free()

        self.qubit_layout = not self.qubit_layout
        self.n_words_per_col = new_words_per_major

        var rt_layout = RuntimeLayout[Self.layout_type].row_major(
            Index(new_major, new_words_per_major)
        )
        self._data = LayoutTensor[
            int_type, Self.layout_type, MutOrigin.external
        ](new_ptr, rt_layout)

    fn is_zero(self) -> Bool:
        var major = self.n_cols if self.qubit_layout else self.n_rows
        var total_words = major * self.n_words_per_col
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

        var major = self.n_cols if self.qubit_layout else self.n_rows
        # Check diagonal
        for i in range(major):
            if not all_good:
                break

            # We want M[r, r] == 1 for all r.
            # In our layout, M[r, c] is bit r of column c.
            # So for column c, we expect bit c to be 1, and all others 0.

            var diag_w = i >> int_bit_exp
            var diag_b = i & (int_bit_width - 1)
            var expected_diag = Scalar[int_type](1) << diag_b

            for w in range(self.n_words_per_col):
                var val = self._data[i, w][0]
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

    fn swap_rows(mut self, r1: Int, r2: Int):
        # transpose if not column-
        if self.qubit_layout:
            self.transpose()

        # Now in Row-Major, rows are contiguous
        @parameter
        fn vec_swap[width: Int](w: Int):
            var val1 = self._data.load[width](Index(r1, w))
            var val2 = self._data.load[width](Index(r2, w))
            self._data.store[width](Index(r1, w), val2)
            self._data.store[width](Index(r2, w), val1)

        vectorize[vec_swap, simd_width](self.n_words_per_col)

    fn xor_row(mut self, target_row: Int, source_row: Int):
        if self.qubit_layout:
            self.transpose()

        # Now in Row-Major, rows are contiguous
        @parameter
        fn vec_xor[width: Int](w: Int):
            var val_src = self._data.load[width](Index(source_row, w))
            var val_tgt = self._data.load[width](Index(target_row, w))
            self._data.store[width](Index(target_row, w), val_tgt ^ val_src)

        vectorize[vec_xor, simd_width](self.n_words_per_col)

    # target_col ^= source_col
    fn xor_col(mut self, target_col: Int, source_col: Int):
        if not self.qubit_layout:
            self.transpose()

        # Now in Col-Major, cols are contiguous
        @parameter
        fn vec_xor[width: Int](w: Int):
            var val_src = self._data.load[width](Index(source_col, w))
            var val_tgt = self._data.load[width](Index(target_col, w))
            self._data.store[width](Index(target_col, w), val_tgt ^ val_src)

        vectorize[vec_xor, simd_width](self.n_words_per_col)

    @always_inline
    fn load_col[
        width: Int
    ](mut self, col: Int, w_idx: Int) -> SIMD[int_type, width]:
        if not self.qubit_layout:
            self.transpose()
        return self._data.load[width](Index(col, w_idx))

    @always_inline
    fn store_col[
        width: Int
    ](mut self, col: Int, w_idx: Int, val: SIMD[int_type, width]):
        if not self.qubit_layout:
            self.transpose()
        self._data.store[width](Index(col, w_idx), val)

    @always_inline
    fn load_row[
        width: Int
    ](mut self, row: Int, w_idx: Int) -> SIMD[int_type, width]:
        if self.qubit_layout:
            self.transpose()
        return self._data.load[width](Index(row, w_idx))

    @always_inline
    fn store_row[
        width: Int
    ](mut self, row: Int, w_idx: Int, val: SIMD[int_type, width]):
        if self.qubit_layout:
            self.transpose()
        self._data.store[width](Index(row, w_idx), val)
