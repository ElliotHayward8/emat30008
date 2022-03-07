import matplotlib.pyplot as plt
import numpy as np
from numerical_shooting import find_shooting_orbit


def normal_hopf(u0, t, vars):
    """
    Function which defines the Hopf bifurcation normal form system of ODEs
    :param u0: Parameter values (u1, u2)
    :param t: Time value
    :param vars: Additional variables which are required to define the system of ODEs
    :return: returns an array of du1/dt and du2/dt at (X, t) as a numpy array
    """
    beta, sigma = vars[0], vars[1]
    u1, u2 = u0[0], u0[1]

    du1dt = beta * u1 - u2 + (sigma * u1) * (u1 ** 2 + u2 ** 2)
    du2dt = u1 + beta * u2 + (sigma * u2) * (u1 ** 2 + u2 ** 2)
    return np.array([du1dt, du2dt])


# phase condition for the normal hopf bifurcation
def pc_normal_hopf(u0, vars):
    return normal_hopf(u0, 1, vars)[0]


def modified_hopf(u0, t, vars):
    """
    Function which defines the Hopf bifurcation normal form system of ODEs
    :param u0: Parameter values (u1, u2)
    :param t: Time value
    :param vars: Additional variables which are required to define the system of ODEs
    :return: returns an array of du1/dt and du2/dt at (X, t) as a numpy array
    """
    beta = vars[0]
    u1, u2 = u0[0], u0[1]

    du1dt = (beta * u1) - u2 + u1 * (u1 ** 2 + u2 ** 2) - u1 * ((u1 ** 2 + u2 ** 2) ** 2)
    du2dt = u1 + (beta * u2) + u2 * (u1 ** 2 + u2 ** 2) - u2 * ((u1 ** 2 + u2 ** 2) ** 2)
    return np.array([du1dt, du2dt])


def continuation(f, u0, max_steps, step_size):
    """
    Function which performs natural parameter continuation on an inputted ODE, f
    :param f: An ODE to perform natural parameter continuation on
    :param u0: The intitial state of the system
    :param max_steps: Maximum number of steps to take
    :param step_size: Size of each step
    :return:
    """


def main():
    u0T = [0, 0, 0]



if __name__ == '__main__':
    main()
