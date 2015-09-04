from IPython.display import display

from sympy import \
    Symbol, var, Function, simplify, oo, exp, Eq, \
    Poly, lcm, LC, degree, Integral, integrate, \
    Matrix, BlockMatrix, eye, zeros, \
    latex, ShapeError, ImmutableMatrix, MutableMatrix

# import scipy for numercial results
from scipy.linalg import expm
from scipy.integrate import quad

import numpy as np

from . import utils as utl

__all__ = ['StateSpaceModel', 'TransferFunctionModel']


class StateSpaceModel:
    """state space model (ssm) of a linear, time invariant control system

    Represents the standard state-space model with state matrix A, input matrix B, output matrix C, and
    transmission matrix D. This makes the linear controll system:
        (1) x'(t) = A * x(t) + B * u(t);    x in R^n , u in R^k
        (2) y(t)  = C * x(t) + D * u(t);    y in R^m
    where u(t)  is any input signal, y(t) the corresponding output, and x(t) the systems state.

    Parameters
    ==========

    arg : TransferFunctionModel, List of Sympy-Matrix
        tfm to construct the state space model from, or the Matrices A,B,C,D in a list

    See Also
    ========

    TranferFunctionModel: transfer function model of a lti system
    Utils: mixed matrix and polynomial tools

    References
    ==========

    Joao P. Hespanha, Linear Systems Theory. 2009.
    """

    def __init__(self, arg):

        if isinstance(arg, TransferFunctionModel):

            # call the private method for realization finding
            self.represent = self._find_realization(arg.G, arg.s)

            # create a block matrix [[A,B], [C,D]] for visual representation
            self.BlockRepresent = BlockMatrix([[self.represent[0], self.represent[1]],
                                               [self.represent[2], self.represent[3]]])
            return None

        else:
            # store the argument as representation of the system
            try:
                self.represent = arg[:4]
            except TypeError:
                raise TypeError("'repesentation' must be a list-like object")

            try:
                # assert that A,B,C,D have matching shapes
                if not ((self.represent[0].shape[0] == self.represent[1].shape[0]) and
                        (self.represent[0].shape[1] == self.represent[2].shape[1]) and
                        (self.represent[1].shape[1] == self.represent[3].shape[1]) and
                        (self.represent[2].shape[0] == self.represent[3].shape[0])):
                    raise ShapeError("Shapes of A,B,C,D must fit")

                # create a block matrix [[A,B], [C,D]] for visual representation
                self.BlockRepresent = BlockMatrix([[self.represent[0], self.represent[1]],
                                                   [self.represent[2], self.represent[3]]])
                return None

            except TypeError:
                raise TypeError("entries of 'representation' must be matrices")
            except AttributeError:
                raise TypeError("entries of 'representation' must be matrices")
            except IndexError:
                raise TypeError("'representation' must have at least 4 matrix-valued entries")

    def _find_realization(self, G, s):
        """ Represenatation [A, B, C, D] of the state space model

        Returns the representation in state space of a given transfer function

        Parameters
        ==========

        G: Matrix
            Matrix valued transfer function G(s) in laplace space
        s: symbol
            variable s, where G is dependent from

        See Also
        ========

        Utils : some quick tools for matrix polynomials

        References
        ==========

        Joao P. Hespanha, Linear Systems Theory. 2009.
        """

        A, B, C, D = 4 * [None]

        try:
            m, k = G.shape

        except AttributeError:
            raise TypeError("G must be a matrix")

        # test if G is proper
        if not utl.is_proper(G, s, strict=False):
            raise ValueError("G must be proper!")

        # define D as the limit of G for s to infinity
        D = G.limit(s, oo)

        # define G_sp as the (stricly proper) difference of G and D
        G_sp = simplify(G - D)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # get the coefficients of the monic least common denominator of all entries of G_sp
        # compute a least common denominator using utl and lcm
        lcd = lcm(utl.fraction_list(G_sp, only_denoms=True))

        # make it monic
        lcd = simplify(lcd / LC(lcd, s))

        # and get a coefficient list of its monic. The [1:] cuts the LC away (thats a one)
        lcd_coeff = Poly(lcd, s).all_coeffs()[1:]

        # get the degree of the lcd
        lcd_deg = degree(lcd, s)

        # get the Matrix Valued Coeffs of G_sp in G_sp = 1/lcd * (N_1 * s**(n-1) + N_2 * s**(n-2) .. +N_n)
        G_sp_coeff = utl.matrix_coeff(simplify(G_sp * lcd), s)
        G_sp_coeff = [zeros(m, k)] * (lcd_deg - len(G_sp_coeff)) + G_sp_coeff

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        # now store A, B, C, D in terms of the coefficients of lcd and G_sp
        # define A
        A = (-1) * lcd_coeff[0] * eye(k)

        for alpha in lcd_coeff[1:]:
            A = A.row_join((-1) * alpha * eye(k))

        for i in xrange(lcd_deg - 1):
            if i == 0:
                tmp = eye(k)
            else:
                tmp = zeros(k)

            for j in range(lcd_deg)[1:]:
                if j == i:
                    tmp = tmp.row_join(eye(k))
                else:
                    tmp = tmp.row_join(zeros(k))
            if tmp is not None:
                A = A.col_join(tmp)

        # define B
        B = eye(k)
        for i in xrange(lcd_deg - 1):
            B = B.col_join(zeros(k))

        # define C
        C = G_sp_coeff[0]
        for i in range(lcd_deg)[1:]:
            C = C.row_join(G_sp_coeff[i])

        # return the state space representation
        return [simplify(A), simplify(B), simplify(C), simplify(D)]

    #
    # evaluate(self, u, t)
    #
    def evaluate(self, u, x0, t, t0=0, method=None, return_pretty=False, do_integrals=True, dps=5):
        """evaluate the system output for an input u

        The output of the system y for the output u if given by solving the state equation for x
        and than substituting that into the output equation

        Parameters
        ==========

        u  : one-column matrix
            The input vector in time-space
        x0 : one-column matrix
            the state of the system at time t0
        t  : symbol, tuple (t,[list of times])
            if t is only a symbol, the system is evaluated simbolycaly.
            if t is a tuple of a symbol and a list, the symstem is evaluated numericaly, at the given times in the list
        t0 = 0 : number
            the time t0 at which the state of the system is known

        method : Bool
            not supported yet, always uses diagonalizaton
        return_pretty : Bool
            if True, the funtion returns a tuple of equations showing the input, initial conditions, and the output
        do_integrals : Bool
            if True, the function tries to evaluate the integrals in the solution. if False, it returns an
            Integral object instead. Only valid for symbolic solutions, ignored otherwise
        dps : integer
            the decimal precision of numericial integration

        References
        ==========

        Joao P. Hespanha, Linear Systems Theory. 2009.
        """
        try:
            # assert right shape of u
            if not u.shape[1] == 1:
                raise ShapeError("u must not have more that one column, but has shape", u.shape)

            if not self.represent[3].shape[1] == u.shape[0]:
                raise ShapeError("u must have length", self.represent[3].shape[1])

            # assert right shape of x0
            if not x0.shape[1] == 1:
                raise ShapeError("x0 must not have more than one column, but has shape", x0.shape)

            if not self.represent[0].shape[1] == x0.shape[0]:
                raise ShapeError("x0 must have length", self.represent[0].shape[1])

        #
        # Error handling
        #
        # if .shape goes wrong, a AttributeError is thorwn
        except AttributeError:
            raise TypeError("u and x0 must be matrices!")

        #
        # find out if t is symbol, tuple or given wrong and call subroutines accordingly to that
        #
        sol = None

        try:

            # if t symobl, then calculate the solution symbolicaly
            if isinstance(t, Symbol):
                print "trying to solve now .."
                sol = self._solve_symbolicaly(u, x0, t, t0, do_integrals=do_integrals)

            # if not, try if it is tuple, list or sth.
            elif isinstance(t[0], Symbol):
                # if t[1] is a direct subclass of tuple or list
                if isinstance(t[1], (list, tuple)):

                    # use the private member function of the class to compute the numervial result
                    sol = self._solve_numericaly(u, x0, t[0], t[1], t0, dps=dps)

                #  if its not, try to convert it
                else:
                    sol = self._solve_numericaly(u, x0, t[0], list(t[1]), t0, dps=dps)

        #
        # Error handling
        #
        # index error can occure if t is not list-like
        except IndexError:
                IndexError("t must be symbol or have at least 2 entries")

        # if the conversion goes wrong, its (hopefully) a TypeError
        except TypeError:
                TypeError("t[1] must be list, or list(t[1]) must work")

        #
        # if that worked, return the prestored solution
        #

        if return_pretty is True:

            y = Function('y')
            u_ = Function('u')
            x_ = Function('x')

            return (Eq(u_(t), u), Eq(x_(0), x0), Eq(y(t), sol))

        else:
            return sol

    #
    # _solve_numericaly
    #
    def _solve_numericaly(self, u, x0, t, t_list, t0, dps=2):
        """ returns the numeric evaluation of the system for input u, know state x0 at time t0 and times t_list
        """
        result = []
        for t_i in t_list:
            print t_i,
            # we use the arbitrary precision module mpmath for numercial evaluation of the matrix exponentials
            first = np.array(np.array(self.represent[2]), np.float).dot(
                expm(np.array(np.array((self.represent[0] * (t_i - t0)).evalf()), np.float))
            ).dot(
                np.array(np.array(x0), np.float)
            )

            second = np.array(np.array((self.represent[3] * u.subs(t, t_i)).evalf()), np.float)

            integrand = lambda tau: \
                np.array(np.array(self.represent[2]), np.float).dot(
                    expm(np.array(np.array((self.represent[0] * (t_i - tau)).evalf()), np.float))
                ).dot(
                    np.array(np.array(self.represent[1]), np.float)
                ).dot(
                    np.array(np.array(u.subs(t, tau).evalf()), np.float)
                )

            # the result must have the same shape as D:
            integral = zeros(self.represent[2].rows, 1)

            # Loop through every entry and evaluate the integral using mpmath.quad()
            for row_idx in xrange(self.represent[2].rows):

                    integral[row_idx, 0] = quad(lambda x: integrand(x)[row_idx, 0], t0, t_i)[0]

            result.append(Matrix(first) + Matrix(second) + integral)

        # return sum of results
        return result

    #
    # _solve_symbolicaly
    #
    def _solve_symbolicaly(self, u, x0, t, t0, exp_method="diagonalize", do_integrals=True):
        """ returns the symbolic evaluation of the system for input u and known state x0 at time t0
        """
        # set the valid methods for the matrix exponential
        # TODO: Laplace Transform
        valid_methods = ("diagonalize")

        if exp_method not in valid_methods:
            raise ValueError("unknown method for matrix exponential:", exp_method)

        # define temporary symbols tau
        tau = Symbol('tau', positive=True)
        x = Symbol('x')

        # compute the two matrix exponentials that are used in the general solution
        # to avoid two eigenvalue problems, first solve for a general real x and substitude then
        expAx = simplify(exp(self.represent[0] * x))
        expA = simplify(expAx.subs(x, t - t0))
        expAt = simplify(expAx.subs(x, t - tau))

        # define the integral and heuristic simplification nowing that in the integral, tau < t always holds
        integrand = simplify(self.represent[2] * expAt * self.represent[1] * u.subs(t, tau))
        integrand = simplify(integrand.subs([(abs(t - tau), t - tau), (abs(tau - t), t - tau)]))
        integral = zeros(integrand.shape[0], integrand.shape[1])

        for col_idx in xrange(integrand.cols):

            for row_idx in xrange(integrand.rows):
                try:
                    if not integrand[row_idx, col_idx] == 0:
                        if do_integrals is True:
                            integral[row_idx, col_idx] = simplify(integrate(integrand[row_idx, col_idx], (tau, t0, t)))
                        else:
                            integral[row_idx, col_idx] = Integral(integrand, (tau, t0, t))
                except:
                    integral[row_idx, col_idx] = Integral(integrand, (tau, t0, t))

        # return the general solution
        return simplify(self.represent[2] * expA * x0 + self.represent[3] * u + integral)

    def cotrollability_matrix(self):
        """ Returns the controllability matrix of the system:
            C = [B, A * B, A^2 * B, .. , A^(n-1), B]; A in R^(n x n), B in^R^(n x m)
        """
        res = self.represent[1]
        for i in xrange(self.represent[0].shape[0] - 1):
            res.col_join(self.represent[0] ** i * self.represent[1])
        return res

    def controllable_subspace(self):
        """ Returns a list of vectors that span the controllable subspace of the system.

        This subspace consists of the states x0 for which there exists an input u : [t0, t1] -> R^k, that
        transfers the state x(t0) = x0 to x(t1) = 0.

        The controllable subspace of an lti system is equal to the image of its controllability matrix.
        """
        return self.controllability_matrix().columnspace()

    def is_controllable(self):
        """ Returns True, if the system is controllable.

        A lti system is called 'controllable' if the controllable subspace of the system equals the
        whole state space R^n. This means, that every state x0 can be transfered to zero at any time.

        The package implements the Eigenvector test for controllability
        """
        for eigenvect_of_A_tr in self.represent[0].transpose().eigenvects():
            for idx in xrange(eigenvect_of_A_tr[1]):
                if (self.represent[1] * eigenvect_of_A_tr[2][idx]).is_zero:
                    return False
        return True

    def cascade(self, anotherSystem):
        """ Returns the cascade interconnection of the system and another system

        The casade interconnection of two systems P1 and P2 is the system for which
        u = u1, y = y2 and z = u2 = y1 so that:

               ----    z	 ----
        u --> | P1 | -----> | P2 | --> y
               ----          ----
        """
        if not isinstance(anotherSystem, StateSpaceModel):
            raise TypeError("Argument must be of type StateSpaceModel")
        # assert matching shapes
        if not self.represent[2].shape[0] == anotherSystem.represent[1].shape[1]:
            raise ShapeError("Dimensions of the input of the argument and the ouput of the System must match!")

        newA = self.represent[0].row_join(
            zeros(self.represent[0].rows, anotherSystem.represent[0].cols)
        ).col_join(
            (anotherSystem.represent[1] * self.represent[2]).row_join(anotherSystem.represent[0])
        )
        newB = self.represent[1].col_join(anotherSystem.represent[1] * self.represent[3])
        newC = (anotherSystem.represent[3] * self.represent[2]).row_join(anotherSystem.represent[2])
        newD = anotherSystem.represent[3] * self.represent[3]

        return StateSpaceModel([newA, newB, newC, newD])

    def parallel(self, anotherSystem):
        """ Returns the parallel interconnection of the system and another system

        The parallel interconnection of two systems P1 and P2 is the system for which
        u = u1 + u2 and y = y1 + y2 so that:

                  ----  y1
             --> | P1 |---
            |     ----    |+
        u --|             o ---> y
            |     ----    |+
             --> | P2 |---
                  ----  y2
        """
        if not isinstance(anotherSystem, StateSpaceModel):
            raise TypeError("Argument must be of type StateSpaceModel")
        # assert matching shapes
        if not ((self.represent[1].shape[1] == anotherSystem.represent[1].shape[1]) and
                (self.represent[2].shape[0] == anotherSystem.represent[2].shape[0])):
            raise ShapeError("Dimensions of inputs and outputs must match!")

        newA = self.represent[0].col_join(zeros(self.represent[0].rows, anotherSystem.represent[0].cols)) \
                                .row_join(
                                    zeros(anotherSystem.represent[0].rows, self.represent[0].cols)
                                    .col_join(anotherSystem.represent[0]))
        newB = self.represent[1].col_join(anotherSystem.represent[1])
        newC = self.represent[2].row_join(anotherSystem.represent[2])
        newD = self.represent[3] + anotherSystem.represent[3]

        return StateSpaceModel([newA, newB, newC, newD])

    #
    # define a magic function for unknown method handling
    #   the class tries to pass the method to the matrices in self.represent
    def __getattr__(self, name):

        # dont overwrite private or magic functions!
        if name[0] == '_':
            raise AttributeError("%r object has no attribute %r" %
                                 (self.__class__, name))

        try:
            def handler(*args, **kwargs):

                new_represent = []
                for r in self.represent:
                    methodToCall = getattr(r, name)
                    new_represent.append(methodToCall(*args, **kwargs))
                return StateSpaceModel(new_represent)

        except AttributeError:
            raise AttributeError("%r object has no attribute %r" %
                                 (self.__class__, name))

        return handler

    #
    # _repr_latex_(self)
    #   defines the representation of the class in ipython pretty printing
    #
    def _repr_latex_(self):
        return '$' + latex(self.BlockRepresent) + '$'


class TransferFunctionModel:
    """ Transfer function model of a linear, time invariant crontrol system

    Represents the transfere Function model with a transfer function Matrix G in laplace space.
    The input-output relation for the system in laplace space is then given by:
        y(s) = G(s) * u(s);     s in C
    where u(s) is the input of the system in laplace space and y(s) the corresponding output

    Parameters
    ==========

    arg : StateSpaceModel, Matrix
        the state space model to contruct the transfer function model from, or the transfer matrix G
    s = None : Symbol
        the variable G is dependent from. only has to be set if arg is a non-constant matrix

    See Also
    ========

    TranferFunctionModel: transfer function model of a lti system
    Utils: mixed matrix and polynomial tools

    References
    ==========

    Joao P. Hespanha, Linear Systems Theory. 2009.
    """

    def __init__(self, arg, s=None):

        # check if a variable is given, if not create a new one as class-wide variable
        if s:
            self.s = s
        else:
            self.s = var('s')

        # constructor from a given state space model
        if isinstance(arg, StateSpaceModel):

            try:
                # define G as transfer function for the given state space model via the definition
                self.G = arg.represent[2] * \
                    (self.s * eye(arg.represent[0].shape[0]) - arg.represent[0]).inv() * \
                    arg.represent[1] + arg.represent[3]

                # try to simplify
                self.G = simplify(self.G)

            except ValueError as err:
                raise ValueError(err.args, "Matrix (s*I -A) must be invertible")
            except AttributeError:
                raise TypeError("Only explicit Matrix Type supported for A,B,C,D (.inv() must work)")

        # constrcutor from a given transfer function
        elif isinstance(arg, (Matrix, ImmutableMatrix, MutableMatrix)):

            # set the given transfer function as self.G
            self.G = simplify(arg)

        else:
            raise TypeError("argument of unsupported type")

    #
    # _repr_latex_(self)
    #   defines the representation of the class in ipython pretty printing
    #
    def _repr_latex_(self):
        return '$' + latex(self.G) + '$'

    #
    # evaluate(self, u, s)
    #
    def evaluate(self, u, s):
        """ evaluate the result for input u

        The input u in laplace state depends on a complex variable s the result y is computed by
            y(s) = G(s) * u(s)

        Parameters
        ==========

        u : one-column matrix
            the input vector u in terms of complex variable s
        s : symbol
            the complex variable s u is dependent from.
        """
        # assert right shape of u
        if not u.shape[1] == 1:
            raise ShapeError("u must be a column vector, not a matrix")
        if not self.G.shape[1] == u.shape[0]:
            raise ShapeError("u must have a length of ", self.G.shape[1])

        # return result
        return self.G.subs(self.s, s) * u

    #
    # define a magic function for unknown method handling
    #   the class tries to pass the method to the matrix self.G
    def __getattr__(self, name):

        # dont overwrite private or magic functions!
        if name[0] == '_':
            raise AttributeError("%r object has no attribute %r" %
                                 (self.__class__, name))

        try:
            def handler(*args, **kwargs):
                methodToCall = getattr(self.G, name)
                return TransferFunctionModel(methodToCall(*args, **kwargs))

        except AttributeError:
            raise AttributeError("%r object has no attribute %r" %
                                 (self.__class__, name))
        return handler
