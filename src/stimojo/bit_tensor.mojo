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
        memset(ptr=ptr, value=0, count=alloc_size)

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
        var word = self._data[word_idx][0]
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
        self._data[word_idx] = current_word

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
    """Bit-packed dense Matrix backed with a LayoutTensor.\n
    Supports runtime transpose+copy to maximize cache locality for column-wide ops and row-wide ops.
    """

    # Default is Column-Major bit-packed layout
    # Shape is (n_cols, n_words_per_col)
    # n_words_per_col = ceil(n_rows / int_bit_width)
    comptime layout_type = Layout.col_major(UNKNOWN_VALUE, UNKNOWN_VALUE)

    var n_rows: Int
    var n_cols: Int
    var n_words_per_col: Int
    var _data: LayoutTensor[int_type, Self.layout_type, MutOrigin.external]
    var column_major: Bool

    fn __init__(out self, rows: Int, cols: Int):
        # Allocate ptr size with words per col aligned up to simd_width
        self.n_rows = rows
        self.n_cols = cols
        self.n_words_per_col = align_up(
            (rows + int_bit_width - 1) // int_bit_width, simd_width
        )
        var alloc_size = cols * self.n_words_per_col

        var ptr = alloc[Scalar[int_type]](alloc_size)
        memset(ptr=ptr, value=0, count=alloc_size)

        # Define Layout at runtime
        var rt_layout = RuntimeLayout[Self.layout_type].col_major(
            Index(cols, self.n_words_per_col)
        )
        # Initializae tensor with layout + data from ptr
        self._data = LayoutTensor[
            int_type, Self.layout_type, MutOrigin.external
        ](ptr, rt_layout)
        self.column_major = True

    fn __copyinit__(out self, other: BitMatrix):
        # TODO: handle case of src/target with different memory layouts.
        self.n_rows = other.n_rows
        self.n_cols = other.n_cols
        self.n_words_per_col = other.n_words_per_col
        self.column_major = other.column_major
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
        self.column_major = other.column_major

    fn __del__(deinit self):
        if self._data.ptr:
            self._data.ptr.free()

    fn __getitem__(self, r: Int, c: Int) -> Bool:
        """Fetches bit state stored at index = (row, col)."""
        # self._data[r, w]: Data is packed along cols (words contain bits for multiple cols of a single row)
        # self._data[c,w]: Data is packed along rows (words contain bits for multiple rows of a single column)
        # Figure out which is the major index and minor index
        var major_idx = c if self.column_major else r
        var minor_idx = r if self.column_major else c

        # Locate word, bit indices
        var word_idx = (
            minor_idx >> int_bit_exp
        )  # eq. word = (r or c) // int_bit_width
        var bit_idx = minor_idx & (
            int_bit_width - 1
        )  # bit_idx = idx % int_bit_width
        var word = self._data[major_idx, word_idx][
            0
        ]  # shift word to index then AND with mask=1
        return ((word >> bit_idx) & Scalar[int_type](1)) == 1

    fn __setitem__(mut self, r: Int, c: Int, val: Bool):
        # self._data[r, w]: Data is packed along cols (words contain bits for multiple cols of a single row)
        # self._data[c,w]: Data is packed along rows (words contain bits for multiple rows of a single column)
        # Figure out which is the major index and minor index
        var major_idx = c if self.column_major else r
        var minor_idx = r if self.column_major else c

        var word_idx = minor_idx >> int_bit_exp
        var bit_idx = minor_idx & (int_bit_width - 1)
        var mask = (
            Scalar[int_type](1) << bit_idx
        )  # mask of 0s except at bit_idx

        var word = self._data[major_idx, word_idx][0]
        if val:
            word |= mask  # set bit to 1 at bit_idx via OR with mask
        else:
            word &= ~mask  # set bit to 0 at bit_idx via AND with NOT mask
        self._data[major_idx, word_idx] = word

    fn __eq__(self, other: BitMatrix) -> Bool:
        if self.n_rows != other.n_rows or self.n_cols != other.n_cols:
            return False

        # If layouts match, we can compare words directly
        if self.column_major == other.column_major:
            var major = self.n_cols if self.column_major else self.n_rows
            for i in range(major):
                for w in range(self.n_words_per_col):
                    if self._data[i, w][0] != other._data[i, w][0]:
                        return False
            return True

        # Otherwise, logical comparison
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self[r, c] != other[r, c]:
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
        # note: Tableau will always require n_cols = n_rows.
        # Below treats general case of n_cols possibly different from n_rows.
        var current_major = self.n_cols if self.column_major else self.n_rows
        var current_minor = self.n_rows if self.column_major else self.n_cols

        # Allocate new data ptr
        var new_major = current_minor
        var new_minor = current_major
        var new_words_per_major = align_up(
            (new_minor + int_bit_width - 1) // int_bit_width, simd_width
        )
        self.n_words_per_col = new_words_per_major
        var new_alloc_size = new_major * new_words_per_major

        var new_ptr = alloc[Scalar[int_type]](new_alloc_size)
        memset(ptr=new_ptr, value=0, count=new_alloc_size)

        # Iterate over major axis
        for major_idx in range(current_major):
            # Iterate over words at a major index
            for word_idx in range(self.n_words_per_col):
                var word = self._data[major_idx, word_idx][0]
                # Iterate over bits in the word
                for bit_idx in range(int_bit_width):
                    # find the minor index of the current bit
                    var minor_idx = (
                        word_idx << int_bit_exp
                    ) + bit_idx  # eq. minor_idx = word_idx * int_bit_width + bit_idx
                    # bound check to ensure the bit at minor_idx isnt' a padding bit
                    if minor_idx < current_minor:
                        if ((word >> bit_idx) & Scalar[int_type](1)) == 1:
                            # take bit from (major_idx, word_idx) and store it at
                            # (minor_idx, dest_word_idx) in new ptr
                            var dest_word_idx = major_idx >> int_bit_exp
                            var dest_bit_idx = major_idx & (int_bit_width - 1)
                            var dest_ptr_idx = (
                                minor_idx * new_words_per_major + dest_word_idx
                            )
                            var val = new_ptr.load(dest_ptr_idx)
                            new_ptr.store(
                                dest_ptr_idx,
                                val | (Scalar[int_type](1) << dest_bit_idx),
                            )

        # Initialize transposed LayoutTensor
        var rt_layout = (
            RuntimeLayout[Self.layout_type]
            .row_major(
                Index(new_major, new_words_per_major)
            ) if self.column_major else RuntimeLayout[Self.layout_type]
            .col_major(Index(new_major, new_words_per_major))
        )

        self._data = LayoutTensor[
            int_type, Self.layout_type, MutOrigin.external
        ](new_ptr, rt_layout)

        # flip layout indicator
        self.column_major = not self.column_major

    fn is_zero(self) -> Bool:
        var major = self.n_cols if self.column_major else self.n_rows
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

        var major = self.n_cols if self.column_major else self.n_rows
        for i in range(major):
            if not all_good:
                break
            for j in range(major):
                if not all_good:
                    break
                if i == j:
                    if not self[i, j]:
                        all_good = False
                        break
                else:
                    if self[i, j]:
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
        @parameter
        fn vec_body[width: Int](w: Int):
            var val1 = self.load_row[width](r1, w)
            var val2 = self.load_row[width](r2, w)
            self.store_row[width](r1, w, val2)
            self.store_row[width](r2, w, val1)

        vectorize[vec_body, simd_width](self.n_words_per_col)

    fn xor_row(mut self, target_row: Int, source_row: Int):
        @parameter
        fn vec_body[width: Int](w: Int):
            var val_src = self.load_row[width](source_row, w)
            var val_tgt = self.load_row[width](target_row, w)
            self.store_row[width](target_row, w, val_tgt ^ val_src)

        vectorize[vec_body, simd_width](self.n_words_per_col)

    fn xor_col(mut self, target_col: Int, source_col: Int):
        @parameter
        fn vec_body[width: Int](w: Int):
            var val_src = self.load_col[width](source_col, w)
            var val_tgt = self.load_col[width](target_col, w)
            self.store_col[width](target_col, w, val_tgt ^ val_src)

        vectorize[vec_body, simd_width](self.n_words_per_col)

    @always_inline
    fn load_col[
        width: Int
    ](mut self, col: Int, w_idx: Int) -> SIMD[int_type, width]:
        if not self.column_major:
            self.transpose()
        return self._data.load[width](Index(col, w_idx))

    @always_inline
    fn store_col[
        width: Int
    ](mut self, col: Int, w_idx: Int, val: SIMD[int_type, width]):
        if not self.column_major:
            self.transpose()
        self._data.store[width](Index(col, w_idx), val)

    @always_inline
    fn load_row[
        width: Int
    ](mut self, row: Int, w_idx: Int) -> SIMD[int_type, width]:
        if self.column_major:
            self.transpose()
        return self._data.load[width](Index(row, w_idx))

    @always_inline
    fn store_row[
        width: Int
    ](mut self, row: Int, w_idx: Int, val: SIMD[int_type, width]):
        if self.column_major:
            self.transpose()
        self._data.store[width](Index(row, w_idx), val)
