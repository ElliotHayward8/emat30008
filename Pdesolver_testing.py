from PDEsolver import pde_solver
import numpy as np
from math import pi


def input_tests():
    """
    Tests the inputs of the PDE Solver function
    """


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
    # input_tests()

    output_tests()


if __name__ == '__main__':
    main()
