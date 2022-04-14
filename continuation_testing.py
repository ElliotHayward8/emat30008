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

    # Test the function runs when all the inputs are correct
    try:
        num_continuation(cubic, 'natural', good_u0_guess_cubic, good_pars, good_end_par, good_index, good_max_steps,
                         lambda x: x, fsolve)
        print('All input values good test : Test Passed')
    except (TypeError, ValueError):
        print('All input values good test : Test Failed')
        failed_input_tests.append('All input values good test')
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
        num_continuation(cubic, 'fakemethod', good_u0_guess_cubic, good_pars, good_end_par, good_index, good_max_steps,
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
        print('Some input tests failed: (see printed below)')
        [print(test) for test in failed_input_tests]
        print('\n---------------------------------------\n')


def output_tests():
    """
    Tests for the outputs of find_shooting_orbit
    """
    failed_output_tests, passed = [], True

    # Print the results of all the output tests
    if passed:
        print('\n---------------------------------------\n')
        print('ALL CONTINUATION OUTPUT TESTS PASSED')
        print('\n---------------------------------------')
    else:
        print('\n---------------------------------------\n')
        print('Some output tests failed: (see printed below)')
        [print(test) for test in failed_output_tests]
        print('\n---------------------------------------')


def main():
    input_tests()

    output_tests()


if __name__ == '__main__':
    main()
