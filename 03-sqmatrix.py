from typing import List, Callable, TypeVar, Tuple, Optional, Union
from functools import reduce
from math import isinf, isnan

from common.utils import randomize_int, fast_power
from matrix.Vector import Vector
from matrix.Matrix import Matrix

T = TypeVar("T")


class SqMatrix(Matrix):
    def __init__(self, content):
        """Set self.dim and self.content, assert that number of rows equal number of columns
        Also initialize superclass properties on self."""
        self.dim = len(content)
        self.content = tuple(tuple(row) for row in content)
        assert all(len(row) == self.dim for row in content)
        super().__init__(content)

    def __repr__(self):
        """Print a square matrix as 'SqMatrix(...)' with one row of content per line,
        and indentation."""
        lines = ",\n        ".join(f"{row}" for row in self.content)
        return f"SqMatrix(({lines}))"

    @staticmethod
    def random(dim: int, randomizer: Callable[[], int] = randomize_int()) -> "SqMatrix":
        """Create and return a new square matrix of the given size, being filled
        with randomly selected elements.  Each such element is the return
        value of the given randomizer function.   randomizer will be called
        once for each entry in the resulting matrix."""
        # CHALLENGE: student must complete the implementation.
        # HINT: goal = 1 line
        return SqMatrix.tabulate(dim, lambda i, j: randomizer())

    @staticmethod
    def identity(dim: int) -> "SqMatrix":
        """Return a square matrix of the given size with 1 (integer)
        along the main diagonal, and 0 (integer) off the main diagonal."""
        # CHALLENGE: student must complete the implementation.
        # HINT: goal = 1 line
        return SqMatrix.tabulate(dim, lambda r, c: 1 if r == c else 0)

    @staticmethod
    def zero(dim: int) -> "SqMatrix":
        """Return a square matrix of the given size consisting entirely
        of 0 (integer)."""
        # CHALLENGE: student must complete the implementation.
        # HINT: goal = 1 line
        return SqMatrix.tabulate(dim, lambda i, j: 0)

    @staticmethod
    def tabulate(dim: int, f: Callable[[int, int], float]) -> "SqMatrix":
        """Create and return a new square matrix of the given size.
        The SqMatrix is filled with the return value of the given function, f.
        The function, f, is called once per entry in the resulting matrix,
        and the return value of f(i,j) is the value of m[i][j]"""
        # CHALLENGE: student must complete the implementation.
        # HINT: goal = 1 line
        return Matrix.tabulate(dim, dim, f)

    @staticmethod
    def diagonal(entries) -> "SqMatrix":
        """Given a sequence of values, return a square matrix with those
        values along the main diagonal.   If the sequence has size n,
        then the resulting matrix has dimension n as well.   The diagonal
        matrix has 0 (integer) off the main diagonal."""
        # CHALLENGE: student must complete the implementation.
        # HINT: goal = 1 line
        raise NotImplementedError()

    def gaussian_elimination_back_substitution(self, v: Vector) -> Optional[Vector]:
        """Adjoin the given column vector to the matrix, self.
        Then perform elementary row operations on the given matrix to reduce
        it to row echelon form.   Then perform back substitution to return
        the vector which solves the system."""
        assert isinstance(v, Vector)
        assert v.dim == self.dim

        # compute an echelon form of the matrix, self.
        # this form has zeros below the main diagonal.
        # and non-zeros on the main diagonal unless the matrix is
        # singular in which case there are zeros on the diagonal.
        ech = self.adjoin_col(v).make_row_echelon()
        if ech is None:
            return None
        assert ech.rows == self.rows
        assert ech.cols == self.cols + 1
        return ech.back_substitution()

    def gauss_jordan_elimination(self, v: Vector) -> Optional[Vector]:
        """Adjoin the given column vector to the matrix, self.
        Then use Gauss Jordan elimination (via elementary row operations)
        to reduce the matrix to diagonal.  This may result in a matrix with
        a zero on the diagonal, in which case None is returned, otherwise
        normalize the diagonal and return the vector which solves the system.
        """
        assert isinstance(v, Vector)
        assert v.dim == self.dim

        diag, _ = self.adjoin_col(v).make_unit_diagonal()
        if diag is None:
            return None
        assert diag.rows == self.rows
        assert diag.cols == self.cols + 1

        return diag.col_vec(v.dim)

    def gauss_jordan_inverse(self) -> Tuple[Optional["SqMatrix"], T]:
        """returns 2-tuple of two values:  (inverse,determinant),
        If the determinant is zero, then inverse=None
        """
        diag, det = self.adjoin_cols(SqMatrix.identity(self.dim)).make_unit_diagonal()
        if diag is None:
            return None, 0
        assert diag.rows == self.rows
        assert diag.cols == 2 * self.cols
        return diag.extract_cols(range(self.dim, 2 * self.dim)), det

    def laplacian_expansion(self, zero: T = 0) -> T:
        """Compute the determinant of the given square matrix using
        Laplacian expansion."""

        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 13 lines

        # # n = m.dim
        # if n == 2
        # return m[0][0] * m[1][1] - m[1][0] * m[0][1]
        # # return sum(m[0][i] * m.suppress_rc(0, i).laplacian_expansion() * (-1) ** i)

    def cramers_rule(self, b: Vector) -> Vector:
        """Return a new vector which solves the system of equations Ax=b,
        where A is the matrix self.   If the determinant of A is 0, then None
        is returned.  The solution is found using Cramer\'s Rule:
        I.e., to compute the k'th component of the returned Vector,
        we replaced the k'th column of a by the column vector b,
        and calculate the determinant of that matrix, then divide by
        the determinant of A.
        """
        assert self.dim == b.dim
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 5 lines
        raise NotImplementedError()

    def power(self, p: int) -> "SqMatrix":
        """Raise the matrix to the p'th power."""
        assert isinstance(p, int)
        assert p >= 0
        # CHALLENGE: student must complete the implementation.
        # HINT: goal <= 6 lines
        assert isinstance(p, int) and p >= 0
        if p == 0:
            return SqMatrix.identity(self.dim)
        if p == 1:
            return self
        half_pow = self.power(p // 2)
        result = half_pow * half_pow
        return result if p % 2 == 0 else result * self
