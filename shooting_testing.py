import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
from ODE_Solvers import euler_step, rk4_step, solve_ode, solve_to
from collections import Counter
from numerical_shooting import find_shooting_orbit

# A program which tests the find_shooting_orbit function


def input_tests():
    """
    Test the response of the find_shooting_orbit function to correct and incorrect inputs
    """

    failed_input_tests, passed = [], True
    good_vars = [1, 0.1, 0.16]
    good_u0T = np.array([0.5, 0.5, 15])

    def right_pred_prey_eq(u0, t, *vars):
        """
        A function which defines the predator prey equations
        :param u0: Vector of initial parameter values (x, y)
        :param t: Time value
        :param vars: Additional variables which define the equation (a, b, d)
        :return: Array of derivatives dx/dt and dy/dt (dxdt, dydt)
        """
        x = u0[0]
        y = u0[1]
        a, b, d = vars[0][0], vars[0][1], vars[0][2]
        dxdt = x * (1 - x) - (a * x * y) / (d + x)
        dydt = b * y * (1 - (y / x))
        return np.array([dxdt, dydt])

    # define an ode with the wrong output type
    def wrong_type_output_ode(u0, t, *vars):
        return 'a string not an array'

    # define an ode with the wrong output size
    def wrong_size_output_ode(u0, t, *vars):
        return np.array([u0[0], u0[0], u0[0]])

    # define the correct phase condition for the predator prey equations
    def right_pred_prey_phase_cond(u0, vars):
        return right_pred_prey_eq(u0, 0, vars)[0]

    # define a phase condition with the wrong shape
    def wrong_shape_prey_phase_cond(u0, *vars):
        return [u0[0], u0[0]]

    # define a phase condition with the wrong output type
    def wrong_type_prey_phase_cond(u0, *vars):
        return 'a string not an array'

    # test it works if inputs are all correct
    try:
        find_shooting_orbit(right_pred_prey_eq, good_u0T, right_pred_prey_phase_cond, good_vars)
        print('Right ODE, pc and IC test : Test Passed')
    except (TypeError, ValueError):
        print('Right ODE, pc and IC test : Test Failed')
        failed_input_tests.append('Right ode, pc and IC test')
        passed = False

    # test the error works correctly if the ODE output is the wrong type
    try:
        find_shooting_orbit(wrong_type_output_ode, good_u0T, right_pred_prey_phase_cond, good_vars)
        print('ODE with wrong output type test : Test Failed')
        failed_input_tests.append('ODE with wrong output type test')
        passed = False
    except TypeError:
        print('ODE with wrong output type test : Test Passed')

    # test the error if the ODE is of the wrong type
    try:
        find_shooting_orbit('a string not a function', good_u0T, right_pred_prey_phase_cond, good_vars)
        print('ODE not a function test : Test Failed')
        failed_input_tests.append('ODE not a function test')
        passed = False
    except TypeError:
        print('ODE not a function test : Test Passed')

    # test the error if an ode with the wrongly sized output is used
    try:
        find_shooting_orbit(wrong_size_output_ode, good_u0T, right_pred_prey_phase_cond, good_vars)
        print('ODE with wrongly sized output test : Test Failed')
        failed_input_tests.append('ODE with wrong sized output test')
        passed = False
    except ValueError:
        print('ODE with wrongly sized output test : Test Passed')

    # test if the phase condition isn't a function
    try:
        find_shooting_orbit(right_pred_prey_eq, good_u0T, 'a string not a function', good_vars)
        print('Phase condition is not a function test : Test Failed')
        failed_input_tests.append('Phase condition is not a function test')
        passed = False
    except TypeError:
        print('Phase condition is not a function test : Test Passed')

    # test if the phase condition has a wrongly shaped output
    try:
        find_shooting_orbit(right_pred_prey_eq, good_u0T, wrong_shape_prey_phase_cond, good_vars)
        print('Phase condition with wrongly sized output test : Test Failed')
        failed_input_tests.append('Phase condition with wrongly sized output test')
        passed = False
    except TypeError:
        print('Phase condition with wrongly sized output test : Test Passed')

    # test if the phase condition has the wrongly sized output
    try:
        find_shooting_orbit(right_pred_prey_eq, good_u0T, wrong_type_prey_phase_cond, good_vars)
        print('Phase condition with output of the wrong type test : Test Failed')
        failed_input_tests.append('Phase condition with wrongly sized output test')
        passed = False
    except TypeError:
        print('Phase condition with output of the wrong type test : Test Passed')

    # Print the results of all the input tests
    if passed:
        print('\n---------------------------------------\n')
        print('ALL SHOOTING INPUT TESTS PASSED')
        print('\n---------------------------------------\n')
    else:
        print('\n---------------------------------------\n')
        print('Some input tests failed: (see below)')
        [print(test) for test in failed_input_tests]
        print('\n---------------------------------------\n')


def output_tests():
    """
    Tests for the outputs of find_shooting_orbit
    """
    failed_output_tests, passed = [], True

    # normal hopf bifurcation function
    def normal_hopf(u0, t, vars):
        """
        Function which defines the Hopf bifurcation normal form system of ODEs
        :param u0: Parameter values (u1, u2)
        :param t: Time value
        :param vars: Additional variables which are required to define the system of ODEs
        :return:
        """
        beta, sigma = vars[0], vars[1]
        u1, u2 = u0[0], u0[1]

        du1dt = beta * u1 - u2 + (sigma * u1) * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 + (sigma * u2) * (u1 ** 2 + u2 ** 2)
        return np.array([du1dt, du2dt])

    # phase condition for the normal hopf bifurcation
    def pc_normal_hopf(u0, vars):
        pc = normal_hopf(u0, 1, vars)[0]
        return pc

    # true solution of the normal hopf bifurcation
    def true_hopf_normal(t, phase, vars):
        beta = vars[0]

        u1 = np.sqrt(beta) * np.cos(t + phase)
        u2 = np.sqrt(beta) * np.sin(t + phase)
        return np.array([u1, u2])

    # Values chosen which are close to the solution
    normal_hopf_u0 = np.array([1.3, 0, 6.1])

    normal_hopf_orbit = find_shooting_orbit(normal_hopf, normal_hopf_u0, pc_normal_hopf, [1, -1])

    shooting_u, T = normal_hopf_orbit[:-1], normal_hopf_orbit[-1]
    true_u = true_hopf_normal(0, T, [1, -1])

    # test if the solution from shooting is close to the true solution
    if np.allclose(true_u, shooting_u):
        print('Supercritical-Hopf bifurcation output test : Test Passed')
    else:
        passed = False
        failed_output_tests.append('Supercritical-Hopf bifurcation output test')
        print('Supercritical-Hopf bifurcation output test : Test Failed')

    # Hopf 3D function
    def hopf_3d(u0, t, vars):
        beta, sigma = vars[0], vars[1]

        u1, u2, u3 = u0[0], u0[1], u0[2]

        du1dt = (beta * u1) - u2 + (sigma * u1) * (u1**2 + u2**2)
        du2dt = u1 + (beta * u2) + (sigma * u2) * (u1**2 + u2**2)
        du3dt = -u3
        return np.array([du1dt, du2dt, du3dt])

    # phase condition for the 3D hopf bifurcation
    def pc_hopf_3d(u0, vars):
        pc = hopf_3d(u0, 1, vars)[0]
        return pc

    # true solution for 3D hopf bifurcation
    def true_hopf_3d(t, phase, vars):
        beta = vars[0]
        u1 = np.sqrt(beta) * np.cos(t + phase)
        u2 = np.sqrt(beta) * np.sin(t + phase)
        u3 = np.exp(-(t + phase))
        return np.array([u1, u2, u3])

    # Values chosen which are close to real solution
    hopf_3d_u0 = np.array([1.3, 0, 1, 6.1])

    hopf_3d_orbit = find_shooting_orbit(hopf_3d, hopf_3d_u0, pc_hopf_3d, [1, -1])
    shooting_u, T = hopf_3d_orbit[:-1], hopf_3d_orbit[-1]

    true_u = true_hopf_3d(10 * T, T, [1, -1])

    if np.allclose(true_u, shooting_u):
        print('3D Hopf bifurcation : Test Passed')
    else:
        passed = False
        failed_output_tests.append('3D Hopf bifurcation output test')
        print('3D Hopf bifurcation : Test Failed')

    # Print the results of all the output tests
    if passed:
        print('\n---------------------------------------\n')
        print('ALL SHOOTING OUTPUT TESTS PASSED')
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
