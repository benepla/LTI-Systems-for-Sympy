from __future__ import print_function, division
from sympy import (
    Matrix, zeros, ImmutableMatrix, MutableMatrix, SparseMatrix, MutableDenseMatrix,
    Poly, degree, simplify
)

__all__ = ['matrix_coeff',
           'matrix_degree',
           'fraction_list',
           'is_proper']

_matrixTypes = (
    Matrix, ImmutableMatrix, MutableMatrix, SparseMatrix, MutableDenseMatrix)


#
# matrix_degree(m)
#
def matrix_degree(m, s):
    """returns the highest degree of any entry in m with respect to s

    Parameters
    ==========

    m: Matrix
        matrix to get degree from
    s: Symbol
        Symbol to get degree from (degree can be ambiguous with multiple coefficients in a expression)
    """
    return max(m.applyfunc(lambda en: degree(en, s)))


#
# matrix_coeff(m)
#
def matrix_coeff(m, s):
    """returns the matrix valued coefficients N_i in m(x) = N_1 * x**(n-1) + N_2 * x**(n-2) + .. + N_deg(m)

    Parameters
    ==========

    m : Matrix
        matrix to get coefficient matrices from
    s :
        symbol to compute coefficient list (coefficients are ambiguous for expressins with multiple symbols)
    """

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
                res[c + m_deg - coeff_deg] += \
                    SparseMatrix(m.shape[0], m.shape[1], {(r, e): 1}) * coeff
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

    if only_denoms is True:
        return map(lambda x: x.as_numer_denom()[1], m)
    if only_numers is True:
        return map(lambda x: x.as_numer_denom()[0], m)
    return map(lambda x: x.as_numer_denom(), m)

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
    if strict is False:
        return all(degree(en.as_numer_denom()[0], s) <=
                   degree(en.as_numer_denom()[1], s) for en in m)
    else:
        return all(degree(en.as_numer_denom()[0], s) <
                   degree(en.as_numer_denom()[1], s) for en in m)
