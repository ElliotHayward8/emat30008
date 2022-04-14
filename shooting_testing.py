import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
from ODE_Solvers import euler_step, rk4_step, solve_ode, solve_to
from collections import Counter
from math import pi
from numerical_shooting import find_shooting_orbit
import warnings

# A program which tests the find_shooting_orbit function


def input_tests():
    """
    Test the response of the find_shooting_orbit function to correct and incorrect inputs
    """

    failed_input_tests, passed = [], True
    good_pars = [1, 0.1, 0.16]
    good_pars, too_small_pars, wrong_type_pars = [1, 0.1, 0.16], [1, 0.1], 'string'
    good_u0T, wrong_size_u0T, wrong_type_u0T = np.array([0.5, 0.5, 15]), np.array([0.5, 0.5]), 'string'

    def right_pred_prey_eq(u0, t, pars):
        """
        A function which defines the predator prey equations
        :param u0: Vector of initial parameter values (x, y)
        :param t: Time value
        :param pars: Additional parameters which define the equation (a, b, d)
        :return: Array of derivatives dx/dt and dy/dt (dxdt, dydt)
        """
        x = u0[0]
        y = u0[1]
        a, b, d = pars[0], pars[1], pars[2]
        dxdt = x * (1 - x) - (a * x * y) / (d + x)
        dydt = b * y * (1 - (y / x))
        return np.array([dxdt, dydt])

    # define an ode with the wrong output type
    def wrong_type_output_ode(u0, t, *pars):
        return 'a string not an array'

    # define an ode with the wrong output size
    def wrong_size_output_ode(u0, t, *pars):
        return np.array([u0[0], u0[0], u0[0]])

    # define the correct phase condition for the predator prey equations
    def right_pred_prey_phase_cond(u0, pars):
        return right_pred_prey_eq(u0, 0, pars)[0]

    # define a phase condition with the wrong shape
    def wrong_shape_prey_phase_cond(u0, *pars):
        return [u0[0], u0[0]]

    # define a phase condition with the wrong output type
    def wrong_type_prey_phase_cond(u0, *pars):
        return 'a string not an array'

    # test it works if inputs are all correct
    try:
        shooting_orbit = find_shooting_orbit(right_pred_prey_eq, good_u0T, right_pred_prey_phase_cond, good_pars)
        print('Right ODE, pc and IC test : Test Passed')
    except (TypeError, ValueError):
        print('Right ODE, pc and IC test : Test Failed')
        failed_input_tests.append('Right ode, pc and IC test')
        passed = False

    # test the error works correctly if the ODE output is the wrong type
    try:
        find_shooting_orbit(wrong_type_output_ode, good_u0T, right_pred_prey_phase_cond, good_pars)
        print('ODE with wrong output type test : Test Failed')
        failed_input_tests.append('ODE with wrong output type test')
        passed = False
    except TypeError:
        print('ODE with wrong output type test : Test Passed')

    # test the error if the ODE is of the wrong type
    try:
        find_shooting_orbit('a string not a function', good_u0T, right_pred_prey_phase_cond, good_pars)
        print('ODE not a function test : Test Failed')
        failed_input_tests.append('ODE not a function test')
        passed = False
    except TypeError:
        print('ODE not a function test : Test Passed')

    # test the error if an ode with the wrongly sized output is used
    try:
        find_shooting_orbit(wrong_size_output_ode, good_u0T, right_pred_prey_phase_cond, good_pars)
        print('ODE with wrongly sized output test : Test Failed')
        failed_input_tests.append('ODE with wrong sized output test')
        passed = False
    except ValueError:
        print('ODE with wrongly sized output test : Test Passed')

    # test for if the phase condition isn't a function
    try:
        find_shooting_orbit(right_pred_prey_eq, good_u0T, 'a string not a function', good_pars)
        print('Phase condition is not a function test : Test Failed')
        failed_input_tests.append('Phase condition is not a function test')
        passed = False
    except TypeError:
        print('Phase condition is not a function test : Test Passed')

    # test if the phase condition has a wrongly shaped output
    try:
        find_shooting_orbit(right_pred_prey_eq, good_u0T, wrong_shape_prey_phase_cond, good_pars)
        print('Phase condition with wrongly sized output test : Test Failed')
        failed_input_tests.append('Phase condition with wrongly sized output test')
        passed = False
    except TypeError:
        print('Phase condition with wrongly sized output test : Test Passed')

    # test if the phase condition has the wrongly sized output
    try:
        find_shooting_orbit(right_pred_prey_eq, good_u0T, wrong_type_prey_phase_cond, good_pars)
        print('Phase condition with output of the wrong type test : Test Failed')
        failed_input_tests.append('Phase condition with wrongly sized output test')
        passed = False
    except TypeError:
        print('Phase condition with output of the wrong type test : Test Passed')

    # test the function if u0T is of the wrong type
    try:
        find_shooting_orbit(right_pred_prey_eq, wrong_type_u0T, right_pred_prey_phase_cond, good_pars)
        print('u0T of the wrong type test : Test Failed')
        failed_input_tests.append('u0T of the wrong type test')
        passed = False
    except TypeError:
        print('u0T of the wrong type test : Test Passed')

    # test the function if u0T is wrongly sized
    try:
        find_shooting_orbit(right_pred_prey_eq, wrong_size_u0T, right_pred_prey_phase_cond, good_pars)
        print('u0T wrongly sized test : Test Failed')
        failed_input_tests.append('u0T wrongly sized test')
        passed = False
    except (IndexError, ValueError):
        print('u0T wrongly sized test : Test Passed')

    # test the function if pars is too small
    try:
        find_shooting_orbit(right_pred_prey_eq, good_u0T, right_pred_prey_phase_cond, too_small_pars)
        print('pars wrongly sized test : Test Failed')
        failed_input_tests.append('pars wrongly sized test')
        passed = False
    except (IndexError, ValueError):
        print('pars wrongly sized test : Test Passed')

    # test the function if pars is of the wrong type
    try:
        find_shooting_orbit(right_pred_prey_eq, good_u0T, right_pred_prey_phase_cond, wrong_type_pars)
        print('pars of the wrong type test : Test Failed')
        failed_input_tests.append('pars of the wrong type test')
        passed = False
    except TypeError:
        print('pars of the wrong type test : Test Passed')

    # Print the results of all the input tests
    if passed:
        print('\n---------------------------------------\n')
        print('ALL SHOOTING INPUT TESTS PASSED')
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

    # normal hopf bifurcation function
    def normal_hopf(u0, t, pars):
        """
        Function which defines the Hopf bifurcation normal form system of ODEs
        :param u0: Parameter values (u1, u2)
        :param t: Time value
        :param pars: Additional parameters which are required to define the system of ODEs
        :return:
        """
        beta, sigma = pars[0], pars[1]
        u1, u2 = u0[0], u0[1]

        du1dt = beta * u1 - u2 + (sigma * u1) * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 + (sigma * u2) * (u1 ** 2 + u2 ** 2)
        return np.array([du1dt, du2dt])

    # phase condition for the normal hopf bifurcation
    def pc_normal_hopf(u0, pars):
        pc = normal_hopf(u0, 1, pars)[0]
        return pc

    # true solution of the normal hopf bifurcation
    def true_hopf_normal(t, phase, pars):
        beta = pars[0]

        u1 = np.sqrt(beta) * np.cos(t + phase)
        u2 = np.sqrt(beta) * np.sin(t + phase)
        return np.array([u1, u2])

    # Values chosen which are close to the solution
    normal_hopf_u0 = np.array([1.3, 0, 6.1])

    normal_hopf_orbit = find_shooting_orbit(normal_hopf, normal_hopf_u0, pc_normal_hopf, [1, -1])
    shooting_u, T = normal_hopf_orbit[:-1], normal_hopf_orbit[-1]
    true_u = true_hopf_normal(T, 0, [1, -1])
    true_T = 2 * pi  # Define the time period of the hopf bifurcation (2pi / 1 = 2pi)

    # test if the solution from shooting is close to the true solution
    if np.allclose(true_u, shooting_u) and np.isclose(T, true_T):
        print('Supercritical-Hopf bifurcation output test : Test Passed')
    else:
        passed = False
        failed_output_tests.append('Supercritical-Hopf bifurcation output test')
        print('Supercritical-Hopf bifurcation output test : Test Failed')

    def pred_prey_eq(X, t, pars):
        """
        A function which defines the predator prey equations
        :param X: Vector of parameter values (x, y)
        :param t: Time value
        :param pars: Additional parameters which define the equation (a, b, d)
        :return: Array of derivatives dx/dt and dy/dt (dxdt, dydt)
        """
        x = X[0]
        y = X[1]
        a, b, d = pars[0], pars[1], pars[2]
        dxdt = x * (1 - x) - (a * x * y) / (d + x)
        dydt = b * y * (1 - (y / x))
        return np.array([dxdt, dydt])

    # define the phase condition for the predator prey equations
    def pred_prey_phase_cond(x0, pars):
        return pred_prey_eq(x0, 0, pars)[0]

    good_pred_prey_u0T = [0.58, 0.285, 21]
    bad_pred_prey_u0T = [50, 50000, 1]

    # Test for bad initial u0 guess
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bad_pred_prey_3d = find_shooting_orbit(pred_prey_eq, bad_pred_prey_u0T, pred_prey_phase_cond, [1, 0.2, 0.1])
        print('Pred-Prey equations with bad u0 test : Test Failed')
        passed = False
        failed_output_tests.append('Pred-Prey equations with bad u0 test')
    except ValueError:
        print('Pred-Prey equations with bad u0 test : Test Passed')

    shooting_orbit = find_shooting_orbit(pred_prey_eq, good_pred_prey_u0T, pred_prey_phase_cond, [1, 0.2, 0.1])

    # True values obtained from inspection of the Pred-Prey graph
    if np.allclose(shooting_orbit, [0.577871, 0.286148, 20.816866]):
        print('Pred-Prey accurate output test : Test Passed')
    else:
        passed = False
        failed_output_tests.append('Pred-Prey accurate output test')
        print('Pred-Prey accurate output test : Test Failed')

    def hopf_3d(u0, t, pars):
        """
        A function which defines the 3D hopf bifurcation
        :param u0: Parameter values (u1, u2, u3)
        :param t: Time value
        :param pars: Additional parameters which define the system of ODEs (beta, sigma)
        :return:
        """
        beta, sigma = pars[0], pars[1]

        u1, u2, u3 = u0[0], u0[1], u0[2]

        du1dt = (beta * u1) - u2 + (sigma * u1) * (u1**2 + u2**2)
        du2dt = u1 + (beta * u2) + (sigma * u2) * (u1**2 + u2**2)
        du3dt = -u3
        return np.array([du1dt, du2dt, du3dt])

    # phase condition for the 3D hopf bifurcation
    def pc_hopf_3d(u0, pars):
        pc = hopf_3d(u0, 1, pars)[0]
        return pc

    # true solution for 3D hopf bifurcation
    def true_hopf_3d(t, phase, pars):
        """
        Function which defines the true solution of the 3D hopf bifurcation
        :param t: Time value (10 times the time period from shooting)
        :param phase: The phase shift of the solution (Time period from shooting)
        :param pars: Additional parameter to define the system of ODEs (beta)
        :return: True solution of the 3D hopf bifurcation
        """
        beta = pars[0]
        u1 = np.sqrt(beta) * np.cos(t + phase)
        u2 = np.sqrt(beta) * np.sin(t + phase)
        u3 = np.exp(-(t + phase))
        return np.array([u1, u2, u3])

    # Values chosen which are close to real solution
    hopf_3d_u0 = np.array([1.3, 0, 0.1, 6.1])

    hopf_3d_orbit = find_shooting_orbit(hopf_3d, hopf_3d_u0, pc_hopf_3d, [1, -1])
    shooting_u, T = hopf_3d_orbit[:-1], hopf_3d_orbit[-1]

    true_u = true_hopf_3d(T, 4 * T, [1, -1])
    true_T = 2 * pi  # Define the time period of the hopf bifurcation (2pi / 1 = 2pi)

    if np.allclose(true_u, shooting_u) and np.isclose(true_T, T):
        print('3D Hopf bifurcation accurate output test : Test Passed')
    else:
        passed = False
        failed_output_tests.append('3D Hopf bifurcation accurate output test')
        print('3D Hopf bifurcation accurate output test : Test Failed')

    # Print the results of all the output tests
    if passed:
        print('\n---------------------------------------\n')
        print('ALL SHOOTING OUTPUT TESTS PASSED')
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
