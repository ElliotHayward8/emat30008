import numpy as np
import matplotlib.pyplot as plt
from ODE_Solvers import solve_ode

def input_tests():
    """
    Test the response of the solve_ode function to correct and incorrect inputs
    """

    failed_input_tests, passed = [], True

    good_t = np.linspace(0, 50, 1000)
    good_x0 = [1, 1]
    good_deltat_max = 0.25

    wrong_size_x0 = [1, 1, 1]

    def func2(X, t):
        """
        This function defines the system of ODEs dx/dt = y and dy/dt = -x
        :param X: A vector of parameter values (x, y)
        :param t: Time value
        :return: returns an array of dx/dt and dy/dt at (X, t) as a numpy array
        """

        x = X[0]
        y = X[1]

        dxdt = y
        dydt = -x

        return np.array([dxdt, dydt])

    def wrong_output_ode(x, t):
        return 'Wrong output type'

    def wrong_output_size(x, t):
        return np.array([x, x, x])

    # Test that the function runs the Euler method when it's provided with the correct inputs
    try:
        solve_ode(func2, good_x0, good_t, good_deltat_max, 'euler', True)
        print('All input values good Euler method test : Test Passed')
    except (TypeError, ValueError, NameError):
        failed_input_tests.append('All input values good Euler method test')
        print('All input values good Euler method test : Test Failed')
        passed = False

    # Test that the function runs the RK4 method  when it's provided with the correct inputs
    try:
        solve_ode(func2, good_x0, good_t, good_deltat_max, 'rk4', True)
        print('All input values good RK4 method test : Test Passed')
    except (TypeError, ValueError, NameError):
        failed_input_tests.append('All input values good RK4 method test')
        print('All input values good RK4 method test : Test Failed')
        passed = False

    # Test the function if a wrongly typed function is inputted
    try:
        solve_ode('not an ode', good_x0, good_t, good_deltat_max, 'rk4', True)
        print('ODE of the wrong type test : Test Failed')
        passed = False
        failed_input_tests.append('ODE of the wrong type test')
    except TypeError:
        print('ODE of the wrong type test : Test Passed')

    # Test the function when the ODE has the wrong output type
    try:
        solve_ode(wrong_output_ode, good_x0, good_t, good_deltat_max, 'rk4', True)
        print('ODE output of the wrong type test : Test Failed')
        passed = False
        failed_input_tests.append('ODE output of the wrong type test')
    except TypeError:
        print('ODE output of the wrong type test : Test Passed')

    # Test the function when the ODE has a wrongly sized output
    try:
        solve_ode(wrong_output_size, good_x0, good_t, good_deltat_max, 'rk4', True)
        print('ODE output of the wrong size test : Test Failed')
        passed = False
        failed_input_tests.append('ODE output of the wrong size test')
    except ValueError:
        print('ODE output of the wrong size test : Test Passed')

    # Test the function when x0 is of the wrong type
    try:
        solve_ode(func2, 'wrong type', good_t, good_deltat_max, 'rk4', True)
        print('x0 of the wrong type test : Test Failed')
        passed = False
        failed_input_tests.append('x0 of the wrong type test')
    except TypeError:
        print('x0 of the wrong type test : Test Passed')

    # Test the function when x0 is wrongly sized
    try:
        solve_ode(func2, wrong_size_x0, good_t, good_deltat_max, 'rk4', True)
        print('x0 of the wrong size test : Test Failed')
        passed = False
        failed_input_tests.append('x0 of the wrong size test')
    except ValueError:
        print('x0 of the wrong size test : Test Passed')

    # Test the function when t is of the wrong type
    try:
        solve_ode(func2, good_x0, 'wrong type', good_deltat_max, 'rk4', True)
        print('t of the wrong type test : Test Failed')
        passed = False
        failed_input_tests.append('t of the wrong type test')
    except TypeError:
        print('t of the wrong type test : Test Passed')

    # Test the function when deltat_max is of the wrong type
    try:
        solve_ode(func2, good_x0, good_t, 'wrong type', 'rk4', True)
        print('deltat_max of the wrong type test : Test Failed')
        passed = False
        failed_input_tests.append('deltat_max of the wrong type test')
    except TypeError:
        print('deltat_max of the wrong type test : Test Passed')

    # Test the function if deltat_max is negative
    try:
        solve_ode(func2, good_x0, good_t, -1, 'rk4', True)
        print('deltat_max is negative test : Test Failed')
        passed = False
        failed_input_tests.append('deltat_max is negative test')
    except ValueError:
        print('deltat_max is negative test : Test Passed')

    # Test the function if a solver of the wrong type is inputted
    try:
        solve_ode(func2, good_x0, good_t, good_deltat_max, 1, True)
        print('solver of the wrong type test : Test Failed')
        passed = False
        failed_input_tests.append('solver of the wrong type test')
    except TypeError:
        print('solver of the wrong type test : Test Passed')

    # Test the function if an incorrectly named solver is inputted
    try:
        solve_ode(func2, good_x0, good_t, good_deltat_max, 'wrong_name', True)
        print('Wrongly named solver test : Test Failed')
        passed = False
        failed_input_tests.append('Wrongly named solver test')
    except NameError:
        print('Wrongly named solver test : Test Passed')

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
    Tests for the outputs of the solve_ode function
    """

    failed_output_tests, passed = [], True

    good_t = np.linspace(0, 1, 100)
    x0 = [1, 1]

    def func2(X, t):
        """
        This function defines the system of ODEs dx/dt = y and dy/dt = -x
        :param X: A vector of parameter values (x, y)
        :param t: Time value
        :return: returns an array of dx/dt and dy/dt at (X, t) as a numpy array
        """

        x = X[0]
        y = X[1]

        dxdt = y
        dydt = -x

        return np.array([dxdt, dydt])

    def true_func2(t):
        """
        Function which calculates the true values of the results of the ODEs dx/dt = y and dy/dt = -x
        :param t: Time value
        :return: Returns an array of x and y at time t
        """

        x = np.sin(t) + np.cos(t)
        y = np.cos(t) - np.sin(t)
        return np.array([x, y])

    # Calculate the solution using the Euler method
    euler_sol = solve_ode(func2, x0, good_t, 0.01, 'euler', True)

    # Calculate the solution using the RK4 method
    rk4_sol = solve_ode(func2, x0, good_t, 0.01, 'rk4', True)

    # Calculate the true solution
    true_sol = true_func2(1)

    # Test the answer from the Euler method - requires a lower tolerance to be set as it is less accurate
    if np.allclose(np.array([euler_sol[0][-1], euler_sol[1][-1]]), true_sol, atol=1e-2):
        print('Euler method on d2xdt2 = -x accuracy test : Test Passed')
    else:
        print('Euler method on d2xdt2 = -x accuracy test : Test Failed')
        failed_output_tests.append('Euler method on d2xdt2 = -x accuracy test')
        passed = False

    # Test the answer from the RK4 method
    if np.allclose(np.array([rk4_sol[0][-1], rk4_sol[1][-1]]), true_sol):
        print('RK4 method on d2xdt2 = -x accuracy test : Test Passed')
    else:
        print('RK4 method on d2xdt2 = -x accuracy test : Test Failed')
        failed_output_tests.append('Euler method on d2xdt2 = -x accuracy test')
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
