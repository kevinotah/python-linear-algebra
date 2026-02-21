from typing import Callable, TypeVar, Tuple, Optional, Generic, Union, List
from common.utils import randomize_int, argmax
from matrix.Vector import Vector

T = TypeVar('T')


class Matrix(Generic[T]):

    def __init__(self, content):
        # Set self.rows and self.cols by measuring the given content.
        # Assert that all the rows have the same length.
        # Assert that there is at least one row and one column.
        # If necessary, convert content to a tuple of tuples, to ensure
        # that it is immutable.
        # If number of rows == number of columns, convert to SqMatrix.
        #
        assert len(content) > 0
        assert all(len(row) > 0 for row in content)
        self.rows = len(content)
        self.cols = len(content[0])
        self.content = tuple(tuple(row) for row in content)
        assert all(self.cols == len(row) for row in content)
        if self.rows == self.cols:
            from matrix.SqMatrix import SqMatrix
            self.__class__ = SqMatrix
            self.dim = self.rows

    def __repr__(self):
        # Return a string representation of the matrix including each row
        # on a separate line. E.g.
        # Matrix((1, 2)
        #        (3, 4))

        # TESTED in Matrix_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 2 lines
        new = []
        for row in self.content:
            new.append(str(row))
        return f"Matrix({"\n".join(new)})"

    def format(self, file=True):
        # compute string which is a printed representation of the Matrix,
        # with columns aligned vertically.
        # Either print or return the string depending on the value of file:
        # If file=True, then print it using print(...),
        # If file=False, then print nothing, but return the string.
        # Otherwise, assume file is a valid 2nd argument of print, and
        # print the string using print(str,file=file)
        strs = [[e.__repr__() for e in row]
                for row in self.content]
        max_len = max(len(str) for row in strs
                      for str in row)
        eq_lens = [[" " * (max_len - len(str)) + str
                    for str in row]
                   for row in strs]
        str = "[" + "\n ".join([', '.join(row) for row in eq_lens]) + " ]"
        if file is False:
            return str
        elif file is True:
            print(str)
        else:
            print(str, file=file)

    @staticmethod
    def tabulate(rows: int, cols: int, f: Callable[[int, int], T]) -> 'Matrix':
        # Create and return a new Matrix of the given size.
        # The Matrix is filled with the return value of the given function, f.
        # The function, f, is called once per entry in the resulting matrix,
        # and the return value of f(i,j) is the value of m[i][j]

        # TESTED in Matrix_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 2 lines
        raise NotImplementedError()

    @staticmethod
    def random(rows: int, cols: int, randomizer: Callable[[], T] = randomize_int()) -> 'Matrix[T]':
        # Create and return a new matrix of the given size, being filled
        # with randomly selected elements.  Each such element is the return
        # value of the given randomizer function.   randomizer will be called
        # once for each entry in the resulting matrix.

        # TESTED in Matrix_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal = 1 line
        raise NotImplementedError()

    @staticmethod
    def zero(rows: int, cols: int) -> 'Matrix[T]':
        # Create and return a new matrix with the given number of rows and columns
        # containing all zeros (integer 0)

        # TESTED in Matrix_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 2 lines
        raise NotImplementedError()

    def __getitem__(self, items):
        # Allow an instance of Matrix to be accessed as m[row][col],
        # e.g, m[3][4]"""

        # TESTED in Matrix_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal = 1 line
        raise NotImplementedError()

    def __eq__(self, other) -> bool:
        # Return True if self and other are the same size,
        # and corresponding entries are equal according to ==.
        # return False if other is not a Matrix, or if it fails to have
        # the same size as self or if any of the corresponding
        # entries are different according to ==.

        # TESTED in Matrix_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 9 lines
        raise NotImplementedError()

    def __add__(self, other: 'Matrix') -> 'Matrix':
        # Create and return a new matrix with this the sum of the
        # given matrices.  This is allowed to blindly fail if the sizes
        # are incompatible.

        # TESTED in Matrix_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 3 lines
        raise NotImplementedError()

    def __sub__(self, other: 'Matrix') -> 'Matrix':
        # Create and return a new matrix with this the difference (subtraction) of the
        # given matrices.  This is allowed to blindly fail if the sizes
        # are incompatible.

        # TESTED in Matrix_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 3 lines
        raise NotImplementedError()

    def suppress_rc(self, remove_row: int, remove_col: int) -> 'Matrix':
        # Return a new Matrix having the specified row and one column removed.
        # If the given matrix has m rows and n cols, then the returned
        # matrix has m-1 rows and n-1 cols.
        assert 0 <= remove_row < self.rows
        assert 0 <= remove_col < self.cols
        assert self.rows > 1
        assert self.cols > 1

        # TESTED in laplacian_determinant_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 9 lines
        raise NotImplementedError()

    def replace_col(self, k: int, v: Vector) -> 'Matrix':
        # Return a new Matrix formed by replacing column k of the given Matrix
        # with the contents of the given Vector, v.
        # If the given Matrix and Vector are not size-compatible
        # (required self.rows = v.dim), an error is raised.
        assert isinstance(v, Vector)
        assert self.rows == v.dim

        # TESTED in cramer_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 3 lines
        raise NotImplementedError()

    def scale(self, num: T) -> 'Matrix':
        # Create and return a matrix which is a scalar multiple of self
        # with the given number.

        # TESTED in Matrix_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 3 lines
        raise NotImplementedError()

    def __mul__(self, other: 'Matrix') -> Union['Matrix', Vector]:
        # Return a matrix with is self multiplied by the given object.
        # other may be any of three possible types.
        # If other is a number, then return a scaled matrix (using self.scale(...)).
        # If other is a Vector, then multiply as if multiplying self (on the left)
        # with an nx1 matrix on the right (where n is the dimension of the Vector).
        # If other is a Matrix of the correct dimensions, the return the matrix
        # product.
        # If the dimensions of the given Vector or Matrix are not compatible,
        # explicitly raise a RuntimeError, explaining the incompatibility.
        import numbers
        # Matrix * number -> Matrix
        if isinstance(other, numbers.Number):
            # TESTED in Matrix_test.py
            # CHALLENGE: student must complete the implementation.
            # HINT: goal = 1 line
            raise NotImplementedError()

        if isinstance(other, Matrix) and self.cols == other.rows:
            # TESTED in Matrix_test.py
            # CHALLENGE: student must complete the implementation.
            # HINT: goal <= 4 lines
            raise NotImplementedError()

        # Matrix * Vector -> Vector (if dimensions are correct)
        if isinstance(other, Vector) and self.cols == other.dim:
            # TESTED in Matrix_test.py
            # CHALLENGE: student must complete the implementation.
            # HINT: goal <= 3 lines
            raise NotImplementedError()

        raise RuntimeError(f"cannot multiply {self} by {other}")

    def row_operation(self, s1: T, r1: int, s2: T, r2: int) -> 'Matrix':
        # s1 * row r1 + s2 * row r2 -> r2
        # rows which are not r2 are NOT copied, rather a reference is made
        # to the same tuple.
        assert isinstance(r1, int)
        assert isinstance(r2, int)

        # TESTED in row_operations_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 9 lines
        raise NotImplementedError()

    def scale_row(self, s: T, r: int) -> 'Matrix':
        # s * row r -> row r
        # Rows different from row r are NOT copied,
        # a reference is made to the tuple already existing.
        assert isinstance(r, int)

        # TESTED in row_operations_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 8 lines
        raise NotImplementedError()

    def swap_cols(self, c1: int, c2: int) -> 'Matrix':
        # if c1==c2, return self, otherwise return a new
        # Matrix having the specified columns swapped.

        # TESTED in row_operations_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 4 lines
        raise NotImplementedError()

    def swap_rows(self, r1: int, r2: int) -> 'Matrix':
        # If r1==r2, return self, otherwise return a new matrix
        # with the specified rows swapped.  This is done without allowing
        # new rows, instead a new tuple is formed with the original row-tuples
        # in a different order.
        assert isinstance(r1, int), f"r1 = {r1}"
        assert isinstance(r2, int), f"r2 = {r2}"
        assert 0 <= r1 < self.rows
        assert 0 <= r2 < self.cols

        # TESTED in row_operations_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 11 lines
        raise NotImplementedError()

    def transpose(self) -> 'Matrix':
        # Return a new matrix having rows and columns swapped
        # newmatrix[i][j] == self[j][i]
        
        # TESTED in Matrix_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 3 lines
        raise NotImplementedError()

    def adjoin_col(self, v: Vector) -> 'Matrix':
        # Given a vector, v, return a new matrix with an additional column
        # adjoined to the right-hand-side of the matrix.  The resulting matrix
        # has 1 additional column, but same number of rows as self.
        assert isinstance(v, Vector)
        assert v.dim == self.rows  # this was a BUG, .rows, not .cols

        # TESTED in row_operations_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 3 lines
        raise NotImplementedError()

    def adjoin_cols(self, m: 'Matrix') -> 'Matrix':
        # Given a Matrix, m, return a new matrix with a additional columns
        # adjoined to the right-hand-side of the matrix.  The resulting matrix
        # has additional columns, but same number of rows as self.
        assert isinstance(m, Matrix)
        assert self.rows == m.rows

        # TESTED in row_operations_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal = 1 line
        raise NotImplementedError()

    def extract_rows(self, keep) -> 'Matrix':
        # Create a new Matrix discarding some of the rows and keeping
        # any row, r, for which (r in keep)
        assert len(keep) >= 0
        assert len(keep) <= self.rows

        # TESTED in row_operations_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 3 lines
        raise NotImplementedError()

    def extract_cols(self, keep) -> 'Matrix':
        # Create a new matrix discarding some columns and keeping
        # any column, c, for which (c in keep)
        assert len(keep) >= 0
        assert len(keep) <= self.cols

        # TESTED in row_operations_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 3 lines
        raise NotImplementedError()

    def adjoin_rows(self, m: 'Matrix') -> 'Matrix':
        # Given matrix m, create a new Matrix with the rows of m
        # adjoined below the bottom-most row of self.
        assert isinstance(m, Matrix)
        assert self.cols == m.cols

        # TESTED in row_operations_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal = 1 line
        raise NotImplementedError()

    def adjoin_row(self, v: Vector) -> 'Matrix':
        # Given a Vector, v, interpret it as a row vector, and
        # append the corresponding 1xn matrix (where n is the dimension of
        # the Vector) after the bottom-most row of self.  The resulting
        # Matrix has exactly one more row than self.
        assert isinstance(v, Vector)
        assert v.dim == self.cols  # this was a bug.  .cols not .rows

        # TESTED in row_operations_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 3 lines
        raise NotImplementedError()

    def row_vec(self, r: int) -> Vector:
        # Return a Vector whose elements are the same as the r'th
        # row of self.
        # This function DOES NOT allocate a new tuple, but rather reuses
        # the tuple already existing in the matrix, self.
        assert isinstance(r, int)
        # assert 0 <= r < self.rows

        # TESTED in Matrix_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal = 1 line
        raise NotImplementedError()

    def col_vec(self, c: int) -> Vector:
        # Return a vector whose elements are the same as the c'th
        # column of self.
        assert isinstance(c, int)
        assert 0 <= c < self.cols

        # TESTED in Matrix_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal = 1 line
        raise NotImplementedError()

    @staticmethod
    def cols_to_matrix(vectors: List[Vector]) -> 'Matrix':
        # Take a list of Vectors interpreted as column vectors,
        # and return a Matrix with those as columns
        assert len(vectors) > 0
        assert all(isinstance(v, Vector) for v in vectors)
        assert all(vectors[0].dim == v.dim for v in vectors)

        # TESTED in row_operations_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 3 lines
        raise NotImplementedError()

    @staticmethod
    def rows_to_matrix(vectors: List[Vector]) -> 'Matrix':
        # Take a list of Vectors interpreted as row vectors,
        # and return a Matrix with those as columns.
        # This function does not allocate new tuples to represent
        # the rows, but rather it reuses the existing content
        # of the vectors to form the rows.
        assert len(vectors) > 0
        assert all(isinstance(v, Vector) for v in vectors)
        assert all(vectors[0].dim == v.dim for v in vectors)

        # TESTED in row_operations_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal = 1 line
        raise NotImplementedError()

    def back_substitution(self) -> Vector:
        # Assume self is a matrix which has already been reduced to row echelon form.
        # I.e., there are k rows and k+1 columns (for some k).
        # Return a Vector of dimension self.rows, which is a result of the back-substitution
        # algorithm.
        assert self.cols == self.rows + 1
        dim = self.rows
        x = [0] * dim
        for k in range(dim - 1, -1, -1):
            # x[k] = ???

            # TESTED in gaussian_test.py
            # CHALLENGE: student must complete the implementation.
            # HINT: goal <= 3 lines
            raise NotImplementedError()

        return Vector(x)

    def make_row_echelon(self) -> Optional['Matrix']:
        # Using elementary row operations, reduce the Matrix to having
        # zeros below the main diagonal, and non-zero on the main
        # diagonal.  If this is impossible, return None.
        # Otherwise, return a new matrix of the same dimension as self, but for which
        # there are zeros below the main diagonal, and no zeros on the main diagonal.

        m = self
        for k in range(0, self.rows - 1):
            pivot = m.find_pivot_row(k)
            if pivot is None:
                return None
            else:
                # swap row k with the pivot row
                # m = m.???

                # TESTED in gaussian_test.py
                # CHALLENGE: student must complete the implementation.
                # HINT: goal = 1 line
                raise NotImplementedError()

            for r in range(k + 1, self.rows):
                # row operation to put 0 into row r column k
                # m = m.???

                # TESTED in gaussian_test.py
                # CHALLENGE: student must complete the implementation.
                # HINT: goal = 1 line
                raise NotImplementedError()

        return m

    def make_unit_diagonal(self) -> Tuple[Optional['Matrix'], T]:
        # Using elementary row operations, reduce the Matrix to having
        # zeros above and below the main diagonal.
        # Return a tuple of two values (Matrix, scalar).
        # The Matrix is a new matrix which has zeros above and below the main
        # diagonal from column 0 to column self.rows-1, and 1 on
        # the main diagonal.
        # The scalar is the product of factors associated with the elementary
        # row operations performed.
        # If it is not possible to form a diagonal matrix with non-zero's on the
        # main diagonal then return (None, 0)
        m = self
        det = 1
        # k iterates over the columns,
        #  note that m[k][k] is the element on the main diagonal
        for k in range(0, self.rows):
            pivot = m.find_pivot_row(k)
            if pivot is None:
                return None, 0
            else:
                # m = ???, need two swap row k with the pivot row

                # TESTED in gaussian_test.py
                # CHALLENGE: student must complete the implementation.
                # HINT: goal = 1 line
                raise NotImplementedError()

            assert m[k][k] != 0
            if pivot != k:
                # if we pivoted then we must multiply the determinant by -1

                # TESTED in gaussian_test.py
                # CHALLENGE: student must complete the implementation.
                # HINT: goal = 1 line
                raise NotImplementedError()

            # divide the row by the element on the main diagonal
            #   putting 1 on the main diagonal
            det *= m[k][k]
            m = m.scale_row(1 / m[k][k], k)
            # zero out column k, except for m[k][k]
            # iterate over all the rows, but skip row k
            for r in range(0, self.rows):
                # perform row operations to zero out above and below the diagonal element m[k][k]
                if r != k:
                    det *= -1 / m[k][k]

                    # TESTED in gaussian_test.py
                    # CHALLENGE: student must complete the implementation.
                    # HINT: goal = 1 line
                    raise NotImplementedError()

            # in case there was round off error, we can force 0
            # into all the elements of this column except the
            # element on the diagonal which should be 1.
            m = m.replace_col(k, Vector([1 if k == c else 0
                                         for c in range(m.rows)]))

        return m, det

    def find_pivot_row(self, c: int, epsilon: float = 0.00001):
        # Return the index of the row which has the largest element
        # in column c. i.e. largest in absolute value.   If two elements
        # have the same largest value, the minimum row index is returned.
        # However, if the column has only 0 in every row, then None
        # is returned.
        p = argmax(lambda r: abs(self[r][c]),
                   range(c, self.rows))
        if self[p][c] == 0:
            return None
        else:
            return p

    def norm(self) -> T:
        # Compute the norm (L-1) of the given matrix.  I.e.,
        # absolute value of most-positive (largest) or most-negative
        # (smallest) element.

        # TESTED in Matrix_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 3 lines
        raise NotImplementedError()

    def distance(self, other) -> T:
        # Compute the distance between two matrices using the norm
        # self.norm(...)

        # TESTED in Matrix_test.py
        # CHALLENGE: student must complete the implementation.
        # HINT: goal = 1 line
        raise NotImplementedError()

