from PDEsolver import pde_solver
import numpy as np
from math import pi


def input_tests():
    """
    Tests the inputs of the PDE Solver function
    """
    failed_input_tests, passed = [], True

    right_kappa, L, right_T, right_mt, right_mx = 1.0, 1.0, 0.5, 3000, 10
    right_bc_0, right_bc_L = 0, 0

    def u_i(x, p=1):
        # initial temperature distribution
        y = (np.sin(pi * x / L)) ** p
        return y

    def wrong_output_ic(x):
        return [x, x]

    # Test the function runs when given inputs of the right type
    try:
        pde_solver(u_i, right_mx, right_mt, right_kappa, L, right_T, right_bc_0, right_bc_L)
        print('Correct input PDE solver test : Test Passed')
    except (TypeError, ValueError):
        print('Correct input PDE solver test : Test Failed')
        failed_input_tests.append('Correct input PDE solver test')
        passed = False

    # Test the function when IC isn't a function
    try:
        pde_solver(5, right_mx, right_mt, right_kappa, L, right_T, right_bc_0, right_bc_L)
        print('IC of the wrong type test : Test Failed')
        failed_input_tests.append('IC of the wrong type test')
    except TypeError:
        print('IC of the wrong type test : Test Passed')

    # Test the function when IC gives the wrong output type
    try:
        pde_solver(wrong_output_ic, right_mx, right_mt, right_kappa, L, right_T, right_bc_0, right_bc_L)
        print('IC with wrong output type test : Test Failed')
        failed_input_tests.append('IC with wrong output type test')
    except TypeError:
        print('IC with wrong output type test : Test Passed')

    # Test the function when mx isn't an integer
    try:
        pde_solver(u_i, 5.5, right_mt, right_mx, L, right_T, right_bc_0, right_bc_L)
        print('mx of the wrong type test : Test Failed')
        failed_input_tests.append('mx of the wrong type test')
    except TypeError:
        print('mx of the wrong type test : Test Passed')

    # Test the function when mt isn't an integer
    try:
        pde_solver(u_i, right_mx, 100.5, right_kappa, L, right_T, right_bc_0, right_bc_L)
        print('mt of the wrong type test : Test Failed')
        failed_input_tests.append('mt of the wrong type test')
    except TypeError:
        print('mt of the wrong type test : Test Passed')

    # test the function if L is of the wrong type
    try:
        pde_solver(u_i, right_mx, right_mt, right_kappa, [1, 1], right_T, right_bc_0, right_bc_L)
        print('L of the wrong type test : Test Failed')
        failed_input_tests.append('L of the wrong type test')
    except TypeError:
        print('L of the wrong type test : Test Passed')

    # Print the results of all the input tests
    if passed:
        print('\n---------------------------------------\n')
        print('ALL PDE SOLVER INPUT TESTS PASSED')
        print('\n---------------------------------------\n')
    else:
        print('\n---------------------------------------\n')
        print('Some input tests failed: (see below)')
        [print(test) for test in failed_input_tests]
        print('\n---------------------------------------\n')


def output_tests():
    """
    Tests for the outputs of the PDE Solver function
    """
    failed_output_tests, passed = [], True

    # Set problem parameters/functions
    kappa, L, T, mt = 1.0, 1.0, 0.5, 3000

    def u_i(x, p=1):
        # initial temperature distribution
        y = (np.sin(pi * x / L)) ** p
        return y

    x, u_j = pde_solver(u_i, 10, 3000, 1.0, L, T, 0, 0, 'forward_euler')
    x_fe, u_j_fe = pde_solver(u_i, 10, 3000, 1.0, L, T, 0, 0, 'fe matrix vector')

    if np.allclose(u_j, u_j_fe) and np.allclose(x, x_fe):
        print('Forward Euler and FE matrix vector same output test : Test Passed')
    else:
        passed = False
        failed_output_tests.append('Forward Euler and FE matrix vector same output test')
        print('Forward Euler and FE matrix vector same output test : Test Failed')

    if passed:
        print('\n---------------------------------------\n')
        print('ALL PDE SOLVER OUTPUT TESTS PASSED')
        print('\n---------------------------------------')
    else:
        print('\n---------------------------------------\n')
        print('Some output tests failed: (see below)')
        [print(test) for test in failed_output_tests]
        print('\n---------------------------------------')


def main():
    input_tests()

    output_tests()


if __name__ == '__main__':
    main()
