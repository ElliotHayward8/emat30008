from PDEsolver import pde_solver
import numpy as np
from math import pi


def input_tests():
    """
    Tests the inputs of the PDE Solver function
    """
    failed_input_tests, passed = [], True

    right_kappa, L, right_T, right_mt, right_mx = 1.0, 1.0, 0.5, 3000, 10

    def bc_is_0(x, t):
        return 0

    def bc_is_1(x, t):
        return 1

    def u_i(x, p=1):
        # Initial temperature distribution
        y = (np.sin(pi * x / L)) ** p
        return y

    def wrong_output_ic(x):
        return [x, x]

    # Test the function runs when given inputs of the right type
    try:
        pde_solver(u_i, right_mx, right_mt, right_kappa, L, right_T, bc_is_0, bc_is_0)
        print('Correct input PDE solver test : Test Passed')
    except (TypeError, ValueError):
        print('Correct input PDE solver test : Test Failed')
        failed_input_tests.append('Correct input PDE solver test')
        passed = False

    # Test the forward euler method when the lambda value isn't within [0, 0.5]
    try:
        pde_solver(u_i, right_mx + 100, right_mt, right_kappa, L, right_T, bc_is_0, bc_is_0)
        print('Lambda outside of range test : Test Failed')
        failed_input_tests.append('Lambda outside of range test')
    except ValueError:
        print('Lambda outside of range test : Test Passed')

    # Test the function when IC isn't a function
    try:
        pde_solver(5, right_mx, right_mt, right_kappa, L, right_T, bc_is_0, bc_is_0)
        print('IC of the wrong type test : Test Failed')
        failed_input_tests.append('IC of the wrong type test')
    except TypeError:
        print('IC of the wrong type test : Test Passed')

    # Test the function when IC gives the wrong output type
    try:
        pde_solver(wrong_output_ic, right_mx, right_mt, right_kappa, L, right_T, bc_is_0, bc_is_0)
        print('IC with wrong output type test : Test Failed')
        failed_input_tests.append('IC with wrong output type test')
    except TypeError:
        print('IC with wrong output type test : Test Passed')

    # Test the function when mx isn't an integer
    try:
        pde_solver(u_i, 5.5, right_mt, right_mx, L, right_T, bc_is_0, bc_is_0)
        print('mx of the wrong type test : Test Failed')
        failed_input_tests.append('mx of the wrong type test')
    except TypeError:
        print('mx of the wrong type test : Test Passed')

    # Test the function when mt isn't an integer
    try:
        pde_solver(u_i, right_mx, 100.5, right_kappa, L, right_T, bc_is_0, bc_is_0)
        print('mt of the wrong type test : Test Failed')
        failed_input_tests.append('mt of the wrong type test')
    except TypeError:
        print('mt of the wrong type test : Test Passed')

    # Test the function if kappa is of the wrong type
    try:
        pde_solver(u_i, right_mx, right_mt, [1, 1], L, right_T, bc_is_0, bc_is_0)
        print('kappa of the wrong type test : Test Failed')
        failed_input_tests.append('kappa of the wrong type test')
    except TypeError:
        print('kappa of the wrong type test : Test Passed')

    # Test the function if L is of the wrong type
    try:
        pde_solver(u_i, right_mx, right_mt, right_kappa, [1, 1], right_T, bc_is_0, bc_is_0)
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
        print('Some input tests failed: (see printed below)')
        [print(test) for test in failed_input_tests]
        print('\n---------------------------------------\n')


def output_tests():
    """
    Tests for the outputs of the PDE Solver function
    """
    failed_output_tests, passed = [], True

    # Set problem parameters/functions
    kappa, L, T, mt = 1.0, 1.0, 0.5, 3000

    def bc_is_0(x, t):
        return 0

    def bc_is_1(x, t):
        return 1

    def u_i(x, p=1):
        # Initial temperature distribution
        y = (np.sin(pi * x / L)) ** p
        return y

    x, u_j = pde_solver(u_i, 10, 3000, 1.0, L, T, bc_is_0, bc_is_0, 'dirichlet', 'forward_euler')
    x_fe, u_j_fe = pde_solver(u_i, 10, 3000, 1.0, L, T, bc_is_0, bc_is_0, 'dirichlet', 'fe matrix vector')

    if np.allclose(u_j, u_j_fe) and np.allclose(x, x_fe):
        print('Forward Euler and FE matrix vector same output test : Test Passed')
    else:
        passed = False
        failed_output_tests.append('Forward Euler and FE matrix vector same output test')
        print('Forward Euler and FE matrix vector same output test : Test Failed')

    # Test to see if the function works for Neumann boundary conditions
    L, kappa, T = 1, 0.25, 5
    mx, mt = 400, 400001

    def u_i_neu(x):
        # Initial temperature distribution
        y = 100 * x * (1 - x)
        return y

    def u_neu_exact(x, t):
        y = x - x + (50 / 3)
        return y

    x_fe_pe, u_j_fe_pe = pde_solver(u_i_neu, mx, mt, kappa, L, T, bc_is_0, bc_is_0, 'neumann', 'fe matrix vector')

    true_u_neu = u_neu_exact(x_fe_pe, 5)

    if np.allclose(true_u_neu, u_j_fe_pe):
        print('Forward Euler Neumann BC test : Test Passed')
    else:
        passed = False
        failed_output_tests.append('Forward Euler Neumann BC test test')
        print('Forward Euler Neumann BC test test : Test Failed')

    if passed:
        print('\n---------------------------------------\n')
        print('ALL PDE SOLVER OUTPUT TESTS PASSED')
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
