import warnings
import numpy as np
from numerical_shooting import shooting
from scipy.optimize import fsolve
from scipy.linalg import norm
from value_checks import array_int_or_float
from num_continuation import num_continuation, nat_par_continuation, pseudo_arclength

def input_tests():
    """
    Test the response of the num_continuation function to correct and incorrect inputs
    """
    failed_input_tests, passed = [], True

    def cubic(x, pars):
        """
        This function defines a cubic equation
        :param x: Value of x
        :param pars: Defines the additional parameter c
        :return: returns the value of the cubic equation at x
        """
        c = pars[0]
        return x ** 3 - x + c

    def wrong_output(x, pars):
        return 'wrong output type'

    good_u0_guess_cubic = np.array([1])
    good_pars, good_end_par = [-2], 2
    good_index = 0
    good_max_steps = 200

    u0_hopfnormal = np.array([1.41, 0, 6.28])

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

    # phase condition for the normal Hopf bifurcation (u1 gradient = 0)
    def pc_normal_hopf(u0, pars):
        return normal_hopf(u0, 0, pars)[0]

    # Test the function runs when all the inputs are correct for natural parameter continuation
    try:
        num_continuation(cubic, 'natural', good_u0_guess_cubic, good_pars, good_end_par, good_index, good_max_steps,
                         lambda x: x, fsolve)
        print('All input values good natural parameter continuation test : Test Passed')
    except (TypeError, ValueError):
        print('All input values good natural parameter continuation test : Test Failed')
        failed_input_tests.append('All input values good natural parameter continuation test')
        passed = False

    # Test the function runs when all inputs are correct for pseudo-arclength continuation
    try:
        num_continuation(normal_hopf, 'natural', u0_hopfnormal, [2, -1], -0.4, 0, 100, shooting, fsolve, pc_normal_hopf)
        print('All input values good pseudo-arclength continuation test : Test Passed')
    except (TypeError, ValueError):
        print('All input values good pseudo-arclength continuation test : Test Failed')
        failed_input_tests.append('All input values good pseudo-arclength continuation test')
        passed = False

    # Test for f of the wrong type being inputted
    try:
        num_continuation(5, 'natural', good_u0_guess_cubic, good_pars, good_end_par, good_index, good_max_steps,
                         lambda x: x, fsolve)
        print('Function of the wrong type test : Test Failed')
        failed_input_tests.append('Function of the wrong type test')
        passed = False
    except TypeError:
        print('Function of the wrong type test : Test Passed')

    # Test for f with the wrong output type being inputted
    try:
        num_continuation(wrong_output, 'natural', good_u0_guess_cubic, good_pars, good_end_par, good_index,
                         good_max_steps, lambda x: x, fsolve)
        print('Function with the wrong output type test : Test Failed')
        failed_input_tests.append('Function with the wrong output type test')
        passed = False
    except TypeError:
        print('Function with the wrong output type test : Test Passed')

    # Test if the method inputted is of the wrong type
    try:
        num_continuation(cubic, 5, good_u0_guess_cubic, good_pars, good_end_par, good_index, good_max_steps,
                         lambda x: x, fsolve)
        print('method of the wrong type test : Test Failed')
        failed_input_tests.append('method of the wrong type test')
        passed = False
    except TypeError:
        print('method of the wrong type test : Test Passed')

    # Test if the method inputted doesn't exist
    try:
        num_continuation(cubic, 'fake_method', good_u0_guess_cubic, good_pars, good_end_par, good_index, good_max_steps,
                         lambda x: x, fsolve)
        print('method which doesn\'t exist test : Test Failed')
        failed_input_tests.append('method which doesn\'t exist test')
        passed = False
    except NameError:
        print('method which doesn\'t exist test : Test Passed')

    # Test if the u0 is of the wrong type
    try:
        num_continuation(cubic, 'natural', 'good_u0_guess_cubic', good_pars, good_end_par, good_index, good_max_steps,
                         lambda x: x, fsolve)
        print('u0 of the wrong type test : Test Failed')
        failed_input_tests.append('u0 of the wrong type test')
        passed = False
    except TypeError:
        print('u0 of the wrong type test : Test Passed')

    # Test if pars is of the wrong type
    try:
        num_continuation(cubic, 'natural', good_u0_guess_cubic, 'good_pars', good_end_par, good_index, good_max_steps,
                         lambda x: x, fsolve)
        print('pars of the wrong type test : Test Failed')
        failed_input_tests.append('pars of the wrong type test')
        passed = False
    except TypeError:
        print('pars of the wrong type test : Test Passed')

    # Test if end_par is of the wrong type
    try:
        num_continuation(cubic, 'natural', good_u0_guess_cubic, good_pars, 'good_end_par', good_index, good_max_steps,
                         lambda x: x, fsolve)
        print('end_par of the wrong type test : Test Failed')
        failed_input_tests.append('end_par of the wrong type test')
        passed = False
    except TypeError:
        print('end_par of the wrong type test : Test Passed')

    # Test if index is of the wrong type
    try:
        num_continuation(cubic, 'natural', good_u0_guess_cubic, good_pars, good_end_par, 'good_index', good_max_steps,
                         lambda x: x, fsolve)
        print('index of the wrong type test : Test Failed')
        failed_input_tests.append('index of the wrong type test')
        passed = False
    except TypeError:
        print('index of the wrong type test : Test Passed')

    # Test if index value is a negative integer
    try:
        num_continuation(cubic, 'natural', good_u0_guess_cubic, good_pars, good_end_par, -5, good_max_steps,
                         lambda x: x, fsolve)
        print('index is a negative integer test : Test Failed')
        failed_input_tests.append('index is a negative integer test')
        passed = False
    except ValueError:
        print('index is a negative integer test : Test Passed')

    # Test if the index value is a float rather than an integer
    try:
        num_continuation(cubic, 'natural', good_u0_guess_cubic, good_pars, good_end_par, 5.4, good_max_steps,
                         lambda x: x, fsolve)
        print('index is a float not an integer test : Test Failed')
        failed_input_tests.append('index is a float not an integer test')
        passed = False
    except TypeError:
        print('index is a float not an integer test : Test Passed')

    # Test if the index value is out of the range of tha parameters
    try:
        num_continuation(cubic, 'natural', good_u0_guess_cubic, good_pars, good_end_par, 5, good_max_steps,
                         lambda x: x, fsolve)
        print('index value is out of range test : Test Failed')
        failed_input_tests.append('index value is out of range test')
        passed = False
    except ValueError:
        print('index value is out of range test : Test Passed')

    # Test if max_steps is of the wrong type
    try:
        num_continuation(cubic, 'natural', good_u0_guess_cubic, good_pars, good_end_par, good_index,'good_max_steps',
                         lambda x: x, fsolve)
        print('max_steps of the wrong type test : Test Failed')
        failed_input_tests.append('max_steps of the wrong type test')
        passed = False
    except TypeError:
        print('max_steps of the wrong type test : Test Passed')

    # Test if max_steps is not an integer
    try:
        num_continuation(cubic, 'natural', good_u0_guess_cubic, good_pars, good_end_par, good_index,
                         100.4, lambda x: x, fsolve)
        print('max_steps is a float test : Test Failed')
        failed_input_tests.append('max_steps is a float test')
        passed = False
    except TypeError:
        print('max_steps is a float test : Test Passed')

    # Test if max_steps is a negative integer
    try:
        num_continuation(cubic, 'natural', good_u0_guess_cubic, good_pars, good_end_par, good_index,
                         -200, lambda x: x, fsolve)
        print('max_steps is a negative integer test : Test Failed')
        failed_input_tests.append('max_steps is a negative integer test')
        passed = False
    except ValueError:
        print('max_steps is a negative integer test : Test Passed')

    # Print the results of all the input tests
    if passed:
        print('\n---------------------------------------\n')
        print('ALL CONTINUATION INPUT TESTS PASSED')
        print('\n---------------------------------------\n')
    else:
        print('\n---------------------------------------\n')
        print('Some input tests failed: (failed tests printed below)')
        [print(test) for test in failed_input_tests]
        print('\n---------------------------------------\n')


def output_tests():
    """
    Tests for the outputs of find_shooting_orbit
    """
    failed_output_tests, passed = [], True

    def cubic(x, pars):
        """
        This function defines a cubic equation
        :param x: Value of x
        :param pars: Defines the additional parameter c
        :return: returns the value of the cubic equation at x
        """
        c = pars[0]
        return x ** 3 - x + c

    # Use fsolve to find the true solution of the cubic equation
    def true_cubic(x, pars):
        return fsolve(cubic, x, args=pars)

    u0_cubic = np.array([1])

    np_par_list, np_sol_list = num_continuation(cubic, 'natural', u0_cubic, [-2], 0.3, 0, 200, lambda x: x, fsolve)

    pa_par_list, pa_sol_list = num_continuation(cubic, 'pseudo', u0_cubic, [-2], 0.3, 0, 200, lambda x: x, fsolve)

    count = 0
    np_true_sols = []

    while count < len(np_par_list):
        true_sol = true_cubic(np_sol_list[count], [np_par_list[count]])
        np_true_sols.append(float(true_sol))
        count += 1

    count = 0
    pa_true_sols = []

    while count < len(pa_sol_list):
        true_sol = true_cubic(pa_sol_list[count], [pa_par_list[count]])
        pa_true_sols.append(true_sol)
        count += 1

    np_sol_list = np.ndarray.tolist(np_sol_list)
    count = 0
    while count < len(np_sol_list):
        np_sol_list[count] = float(np_sol_list[count][0])
        count += 1

    if np.allclose(np_true_sols, np_sol_list, atol=1e-6):
        print('natural parameter cubic accuracy test : Test Passed')
    else:
        print('natural parameter cubic accuracy test test : Test Failed')
        failed_output_tests.append('natural parameter cubic accuracy test')
        passed = False

    if np.allclose(pa_true_sols, pa_sol_list, atol=1e-7):
        print('pseudo-arclength cubic accuracy test : Test Passed')
    else:
        print('pseudo-arclength cubic accuracy test test : Test Failed')
        failed_output_tests.append('pseudo-arclength cubic accuracy test')
        passed = False

    u0_hopfnormal = np.array([1.41, 0, 6.28])

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

    # phase condition for the normal Hopf bifurcation (u1 gradient = 0)
    def pc_normal_hopf(u0, pars):
        return normal_hopf(u0, 0, pars)[0]

    # True solution of the normal hopf bifurcation
    def true_hopf_normal(t, phase, pars):
        beta = pars[0]

        u1 = np.sqrt(beta) * np.cos(t + phase)
        u2 = np.sqrt(beta) * np.sin(t + phase)
        return np.array([u1, u2])

    # Test the values calculated are correct before the bifurcation at Beta = 0 is reached
    np_par_list, np_sol_list = num_continuation(normal_hopf, 'natural', u0_hopfnormal, [2, -1], 0.1, 0, 100,
                                                shooting, fsolve, pc_normal_hopf)

    count = 0
    true_sols = []

    while count < len(np_par_list):
        true_sol = true_hopf_normal(0, np_sol_list[count][-1], [np_par_list[count]])
        true_sols.append(true_sol)
        count += 1

    if np.allclose(true_sols, np_sol_list[:, :-1], atol=1e-7):
        print('hopf normal accuracy test : Test Passed')
    else:
        print('hopf normal accuracy test : Test Failed')
        failed_output_tests.append('hopf normal accuracy test')
        passed = False

    # Print the results of all the output tests
    if passed:
        print('\n---------------------------------------\n')
        print('ALL CONTINUATION OUTPUT TESTS PASSED')
        print('\n---------------------------------------')
    else:
        print('\n---------------------------------------\n')
        print('Some output tests failed: (failed tests printed below)')
        [print(test) for test in failed_output_tests]
        print('\n---------------------------------------')


def main():
    input_tests()

    output_tests()


if __name__ == '__main__':
    main()
