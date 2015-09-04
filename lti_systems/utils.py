from sympy import (
    Matrix, zeros, ImmutableMatrix, MutableMatrix, SparseMatrix, MutableDenseMatrix,
    ShapeError, eye, Poly, degree, flatten, Symbol, simplify
)
from sympy.solvers import solve
from sympy.physics.quantum.matrixutils import matrix_tensor_product

from mpmath import mpf, mpc, pi, sin, tan, exp
# from IPython.display import display

_matrixTypes = (
    Matrix, ImmutableMatrix, MutableMatrix, SparseMatrix, MutableDenseMatrix)


#
# matrix_degree(m)
#
def matrix_degree(m, s):
    """returns the highest degree of any entry in m with respect to s"""
    deg = 0
    try:
        for row in m.tolist():
            for entry in row:
                if deg < degree(entry, s):
                    deg = degree(entry, s)

    except AttributeError:
        raise TypeError("m must support m.tolist()!")
    return deg


#
# matrix_unit((i,j),(n,k))
#
def matrix_unit((i, j), (n, k)):
    """returns a i x j matrix where all entries are zero except the (n,k)th, which is one"""
    res = zeros(i, j)
    res[n, k] = 1
    return res


#
# matrix_coeff(m)
#
def matrix_coeff(m, s):
    """returns the matrix valued coefficients N_i in m(x) = N_1 * x**(n-1) + N_2 * x**(n-2) + .. + N_deg(m)"""

    m_deg = matrix_degree(m, s)
    res = [zeros(m.shape[0], m.shape[1])] * (m_deg + 1)

    for r, row in enumerate(m.tolist()):
        for e, entry in enumerate(row):

            entry_coeff_list = Poly(entry, s).all_coeffs()
            if simplify(entry) == 0:
                coeff_deg = 0
            else:
                coeff_deg = degree(entry, s)

            for c, coeff in enumerate(entry_coeff_list):
                res[c + m_deg - coeff_deg] += matrix_unit(m.shape, (r, e)) * coeff
    return res


#
# fraction_list(m)
#
def fraction_list(m, only_denoms=False, only_numers=False):
    """list of fractions of m

    retuns a list of tuples of the numerators and denominators of all entries of m.
    the entries of m can be any sort of expressions.
    result[i*j + j][0/1] is the numerator/denominator of the matrix element m[i,j]

    Parameters
    ==========

    m : Matrix
        the matrix we want the list of fraction from

    Flags
    =====

    only_denoms=False : Bool
        if True, function only returns a list of denominators, not tuples
    only_numers)False: Bool
        if True, function only returns a list of nmerators, not tuples

    """

    if (only_denoms is True) and (only_numers is True):
        raise ValueError(
            "at least one of only_denoms and only_numers must be False")

    res = []
    try:
        for row in m.tolist():
            try:
                res += map(lambda x: x.as_numer_denom(), row)

            except AttributeError:
                raise TypeError("entries of m must be expression-like")

    except AttributeError:
        raise TypeError("type of m must support m.tolist() (eg. Sympy Matrix)")

    if only_denoms is True:
        return map(lambda x: x[1], res)
    if only_numers is True:
        return map(lambda x: x[0], res)
    return res

#
# is_proper(m, s, strict=False)
#


def is_proper(m, s, strict=False):
    """is_proper

    tests if the degree of the numerator does not exceed the degree of the denominator
    for all entries of a given matrix.

    Parameters
    ==========

    m : Matrix
        matrix to test if proper

    Flags
    =====

    strict = False
        if rue, the function returns True only if the degree of the denominator is always greater
        than the degree of the numerator
    """
    res = True
    if strict is False:
        for i in fraction_list(m):
            if not degree(i[0], s) <= degree(i[1], s):
                res = False
    else:
        for i in fraction_list(m):
            if not degree(i[0], s) < degree(i[1], s):
                res = False
    return res


#
# vectorize(M)
#
def vectorize(M, sparse=False):
    """vectorizes the matrix M

    vectorization is a linear transformation which converts a matrix into a column vector. For a m x n matrix M,
    the vectorization vec(M) is the mn x 1 matrix obtained by stacking the columns of the matrix on top of one another

    Parameters
    ==========

    m : Matrix
        matrix to vectorize

    sparse = False
        set True if input is an instance of sparse matrix

    Reference
    =========

    Wikipedia - Vectorization(mathematics)
    https://en.wikipedia.org/wiki/Vectorization_%28mathematics%29
    """
    if not isinstance(M, _matrixTypes):
        raise TypeError("Argument must be matrix")

    if sparse is False:
        vec = []
        for i in xrange(M.cols):
            vec.append(M.col(i))

        return Matrix(flatten(vec))

    elif sparse is True:
        if not isinstance(M, SparseMatrix):
            TypeError("is sparse is True, M must be instance of SparseMatrix")
        vec = SparseMatrix(M.rows * M.cols, 1, {})
        for entrytuple in M.col_list():
            vec[entrytuple[0] + entrytuple[1] * M.rows] = entrytuple[2]

        return vec
    else:
        raise ValueError("'sparse' must be True or False")


#
# inverse_vectorize(v, (n,m))
#
def inverse_vectorize(n, m, v, sparse=False):
    """ computes a matrix M of shape (n, m) from the vector v, so that M = inverse_vectorize(vectorize(M))
    """
    if not isinstance(v, _matrixTypes):
        raise TypeError("v must be one of", _matrixTypes, "but is", type(v))
    if not (v.rows == n * m and v.cols == 1):
        raise ShapeError("v must have shape", (n * m, 1))

    if sparse is False:
        res = zeros(n, m)
        for i in xrange(m):
            res[:, i] = v[i * n: (i + 1) * n]
        return res

    elif sparse is True:
        if not isinstance(v, SparseMatrix):
            TypeError("is sparse is True, M must be instance of SparseMatrix")
        res = SparseMatrix(n, m, {})
        for entrytuple in v.col_list():
            res[entrytuple[0] % n, entrytuple[0] / n] = entrytuple[2]
        return res

    else:
        raise ValueError("'sparse' must be True or False")


#
#
#
def _sparse_matrix_tensor_product(A, B):
    res = SparseMatrix(A.rows * B.rows, A.cols * B.cols, {})

    for A_el in A.row_list():
        for B_el in B.row_list():
            res[A_el[0] * B.rows + B_el[0], A_el[1] * B.cols + B_el[1]] = A_el[2] * B_el[2]

    return res


#
# SylvsterSolve(A, B, C)

def SylvesterSolve(A, B, C, method=None, sparse=False, **kwargs):
    """solves the Sylvester Equation AX + XB = C for X

    A and B must be square matrices of shape n x n and m x m. Then C must have shape n x m and
    X has shape n x m aswell
    """
    if not all(isinstance(m, _matrixTypes) for m in (A, B, C)):
        raise TypeError("Arguments must be matrixes")

    if not A.shape[0] == A.shape[1]:
        raise ShapeError("A must be square!")
    if not B.shape[0] == B.shape[1]:
        raise ShapeError("B must be square!")

    n, m = A.shape[0], B.shape[0]

    if not C.shape == (n, m):
        raise ShapeError("c must have shape", (n, m))

    if sparse is False:
        return inverse_vectorize(
            n, m,
            (
                matrix_tensor_product(
                    eye(m), A) + matrix_tensor_product(B.transpose(), eye(n))
            ).solve(vectorize(C), method=method)
        )
    elif sparse is True:
        if method == 'linear_system':

            print 'create SparseLinearSystem ..'
            toSolve = _sparse_matrix_tensor_product(
                SparseMatrix(eye(m)), A).add(_sparse_matrix_tensor_product(B.transpose(), SparseMatrix(eye(n)))
                                             ).row_join(vectorize(C, sparse=True))

            print 'collect nonzero rows ..'
            nz_sys = []
            cur_row = None
            cur_eq = 0
            symbs = set()
            for el in toSolve.row_list():

                if cur_row is None:
                    cur_row = el[0]

                if not el[0] == cur_row:
                    nz_sys.append(cur_eq)
                    cur_row = el[0]
                    cur_eq = 0

                if el[1] == n * m:
                    cur_eq -= el[2]
                else:
                    cur_eq += el[2] * Symbol('x' + str(el[1]))
                    symbs.add(Symbol('x' + str(el[1])))

            nz_sys.append(cur_eq)

            print 'solve the equation system ..'
            sol = solve(nz_sys, symbs, dict=True, **kwargs)[0]
            res = SparseMatrix(n, m, {})

            print 'assingn solution to returned matrix ..'
            for symb in sol:
                idx = int(symb.name[1:])
                res[idx % n, idx / n] = sol[symb]

            return res

        else:

            return inverse_vectorize(
                n, m,
                _sparse_matrix_tensor_product(
                    SparseMatrix(eye(m)), A).add(_sparse_matrix_tensor_product(B.transpose(), SparseMatrix(eye(n)))
                                                 ).solve(SparseMatrix(vectorize(C, sparse=True)), method=method),
                sparse=True
            )
    else:
        raise ValueError("'sparse' must be True or False!")


def LyapunovSolve(A, C, **kwargs):
    """solves the Lyapunov Equation AX + XA.adjoint() = C for X

    A and C must be square  with both shape n x n. X has shape n x n aswell.

    As the Lyapunov Equation is a special case of the sylvester equation, the function
    curretly serves as a wrapper for the 'SylvesterSolve' function
    """
    return SylvesterSolve(A, A.adjoint(), C, **kwargs)


# testfunction: Laplace-transform of exp(-t)
def F(s):
    return 1.0 / (s + 1.0)


class Talbot(object):
    """computes the inverse Laplace transformation numeraicaly using the Talbot method

    Talbot suggested that the Bromwich line be deformed into a contour that begins
    and ends in the left half plane, i.e., z \to \infty at both ends.
    Due to the exponential factor the integrand decays rapidly
    on such a contour. In such situations the trapezoidal rule converge
    extraordinarily rapidly.
    For example here we compute the inverse transform of F(s) = 1/(s+1) at t = 1

    >>> error = Talbot(1,24)-exp(-1)
    >>> error
    (3.3306690738754696e-015+0j)

    Talbot method is very powerful here we see an error of 3.3e-015
    with only 24 function evaluations

    Created by Fernando Damian Nieuwveldt
    email:fdnieuwveldt@gmail.com
    Date : 25 October 2009

    Adapted to mpmath and classes by Dieter Kadelka
    email: Dieter.Kadelka@kit.edu
    Date : 27 October 2009

    Reference
    L.N.Trefethen, J.A.C.Weideman, and T.Schmelzer. Talbot quadratures
    and rational approximations. BIT. Numerical Mathematics,
    46(3):653 670, 2006.
    """

    def __init__(self, F=F, shift=0.0):
        self.F = F
        # test = Talbot() or test = Talbot(F) initializes with testfunction F

        self.shift = shift
        # Shift contour to the right in case there is a pole on the
        #   positive real axis :
        # Note the contour will not be optimal since it was originally devoloped
        #   for function with singularities on the negative real axis For example
        #   take F(s) = 1/(s-1), it has a pole at s = 1, the contour needs to be
        #   shifted with one unit, i.e shift  = 1.
        # But in the test example no shifting is necessary

        self.N = 24
        # with double precision this constant N seems to best for the testfunction
        #   given. For N = 22 or N = 26 the error is larger (for this special
        #   testfunction).
        # With laplace.py:
        # >>> test.N = 500
        # >>> print test(1) - exp(-1)
        # >>> -2.10032517928e+21
        # Huge (rounding?) error!
        # with mp_laplace.py
        # >>> mp.dps = 100
        # >>> test.N = 500
        # >>> print test(1) - exp(-1)
        # >>> -5.098571435907316903360293189717305540117774982775731009465612344056911792735539092934425236391407436e-64

    def __call__(self, t):

        if t == 0:
            print "ERROR:   Inverse transform can not be calculated for t=0"
            return ("Error")

        # Initiate the stepsize
        h = 2 * pi / self.N

        ans = 0.0
        # parameters from
        # T. Schmelzer, L.N. Trefethen, SIAM J. Numer. Anal. 45 (2007) 558-571
        c1 = mpf('0.5017')
        c2 = mpf('0.6407')
        c3 = mpf('0.6122')
        c4 = mpc('0', '0.2645')

        # The for loop is evaluating the Laplace inversion at each point theta i
        #   which is based on the trapezoidal rule
        for k in range(self.N):
            theta = -pi + (k + 0.5) * h
            z = self.shift + self.N / t * (c1 * theta / tan(c2 * theta) - c3 + c4 * theta)
            dz = self.N / t * (-c1 * c2 * theta / sin(c2 * theta)**2 + c1 / tan(c2 * theta) + c4)
            ans += exp(z * t) * self.F(z) * dz

        return ((h / (2j * pi)) * ans).real 
