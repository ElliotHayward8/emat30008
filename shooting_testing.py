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

    def normal_hopf(u0, t, vars):
        beta, sigma = vars[0], vars[1]
        u1, u2 = u0[0], u0[1]

        du1dt = beta * u1 - u2 + (sigma * u1) * (u1 ** 2 + u2 ** 2)
        du2dt = u1 + beta * u2 + (sigma * u2) * (u1 ** 2 + u2 ** 2)
        return np.array([du1dt, du2dt])

    def pc_normal_hopf(u0, vars):
        p = normal_hopf(u0, 1, vars)[0]
        return p

    def true_hopf_normal(t, phase, vars):
        beta = vars[0]

        u1 = np.sqrt(beta) * np.cos(t + phase)
        u2 = np.sqrt(beta) * np.sin(t + phase)
        return np.array([u1, u2])

    passed = True

    normal_hopf_u0 = np.array([1.4, 0, 6.3])

    normal_hopf_orbit = find_shooting_orbit(normal_hopf, normal_hopf_u0, pc_normal_hopf, [1, -1])

    shooting_u = normal_hopf_orbit[:-1]
    T = normal_hopf_orbit[-1]
    true_u = true_hopf_normal(0, T, [1, -1])


def main():
    output_tests()


if __name__ == '__main__':
       main()
