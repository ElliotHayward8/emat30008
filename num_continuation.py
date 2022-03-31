import matplotlib.pyplot as plt
import numpy as np
from numerical_shooting import shooting
from scipy.optimize import fsolve
from value_checks import array_int_or_float


def cubic(x, pars):
    """
    This function defines a cubic equation
    :param x: Value of x
    :param pars: Defines the additional parameter c
    :return: returns the value of the cubic equation at x
    """
    c = pars[0]
    return x ** 3 - x + c


def normal_hopf(u0, t, pars):
    """
    Function which defines the Hopf bifurcation normal form system of ODEs
    :param u0: Parameter values (u1, u2)
    :param t: Time value
    :param pars: Additional parameters which are required to define the system of ODEs (beta, sigma)
    :return: returns an array of du1/dt and du2/dt at (X, t) as a numpy array
    """
    beta, sigma = pars[0], pars[1]
    u1, u2 = u0[0], u0[1]

    du1dt = beta * u1 - u2 + (sigma * u1) * (u1 ** 2 + u2 ** 2)
    du2dt = u1 + beta * u2 + (sigma * u2) * (u1 ** 2 + u2 ** 2)
    return np.array([du1dt, du2dt])


# phase condition for the normal hopf bifurcation (u1 gradient = 0)
def pc_normal_hopf(u0, t, pars):
    return normal_hopf(u0, pars)[0]


def modified_hopf(u0, t, pars):
    """
    Function which defines the Hopf bifurcation normal form system of ODEs
    :param u0: Parameter values (u1, u2)
    :param t: Time value
    :param pars: Additional variables which are required to define the system of ODEs
    :return: returns an array of du1/dt and du2/dt at (X, t) as a numpy array
    """
    beta = pars[0]
    u1, u2 = u0[0], u0[1]

    du1dt = (beta * u1) - u2 + u1 * (u1 ** 2 + u2 ** 2) - u1 * ((u1 ** 2 + u2 ** 2) ** 2)
    du2dt = u1 + (beta * u2) + u2 * (u1 ** 2 + u2 ** 2) - u2 * ((u1 ** 2 + u2 ** 2) ** 2)
    return np.array([du1dt, du2dt])


def nat_par_continuation(f, u0_guess, pars0, max_par, vary_par, max_steps=100, discretisation=shooting,
                         solver=fsolve, phase_cond='none'):
    """
    Function which performs natural parameter continuation on an inputted ODE, f
    :param f: An ODE to perform natural parameter continuation on
    :param u0_guess: Estimated value of the solution at
    :param pars0: The initial variables
    :param max_par: Maximum value of the varying parameter
    :param vary_par: The index position of the parameter which is varying
    :param max_steps: Maximum number of steps to take
    :param discretisation: The discretisation to use
    :param solver: The solver to use
    :return: A list of values of the varied parameter and a list of solution values
    """

    # Check u0_guess, pars0 only contain integers or floats
    array_int_or_float(u0_guess, 'u0_guess')
    array_int_or_float(pars0, 'pars0')

    # Check that vary_par is 0 or a positive integer
    if vary_par >= 0:
        if not isinstance(vary_par, (int, np.int_)):
            raise TypeError(f"vary_par: {vary_par} is not an integer")
    else:
        raise ValueError(f"vary_par: {vary_par} is < 0")

    # define the minimum value of the parameter, and create a list of values of the varying parameter
    min_par = pars0[vary_par]

    par_list = np.linspace(min_par, max_par, max_steps)

    # if a phase condition is required pass it into the pars so that it can be passed into the solver
    if phase_cond != 'none':
        initial_pars0 = (phase_cond, pars0)
    else:
        initial_pars0 = pars0

    print('nat_par initial_pars = ' + str(initial_pars0))
    print('type of discretisation of function' + str(type(discretisation(f))))

    first_sol = solver(discretisation(f), u0_guess, args=initial_pars0)

    # for par in par_list:


def num_continuation(f, method, u0_guess, pars0, max_par, vary_par, max_steps=100, discretisation=shooting,
                     solver=fsolve, phase_cond='none'):

    if method == 'natural parameter':
        nat_par_continuation()



def main():
    u0_guess_hopfnormal = np.array([1.4, 0, 6.3])

    nat_par_continuation(normal_hopf, u0_guess_hopfnormal, [2, -1], 2, 0, 100, shooting, fsolve, pc_normal_hopf)


if __name__ == '__main__':
    main()
