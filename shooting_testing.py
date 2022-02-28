import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
from ODE_Solvers import euler_step, rk4_step, solve_ode, solve_to
from collections import Counter
from numerical_shooting import find_shooting_orbit

# A program which tests the find_shooting_orbit function


def output_tests():
    """
    Tests the outputs of find_shooting_orbit
    """
    failed_output_tests, passed = [], True

    # normal hopf bifurcation function
    def normal_hopf(u0, t, vars):
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

    # Print the results of all the tests
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
    output_tests()


if __name__ == '__main__':
       main()
