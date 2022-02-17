import collections

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
from ODE_Solvers import euler_step, rk4_step, solve_ode, solve_to
from collections import Counter


def pred_prey_eq(X, t, *vars):
    x = X[0]
    y = X[1]
    a, b, d = vars[0][0], vars[0][1], vars[0][2]
    dxdt = x * (1 - x) - (a * x * y) / (d + x)
    dydt = b * y * (1 - (y / x))
    return np.array([dxdt, dydt])


def compare_b_values(b1, b2):
    """
    A function which produces a graph to compare how the predator prey model changes for different values of b with
    initial conditions: x,y = 0.5 and h = 0.001
    :param b1: First value of b
    :param b2: Second value of b
    """
    t_eval = np.linspace(0, 100, 100)
    deltat_max = 0.001

    vars1, vars2 = [1, b1, 0.1], [1, b2, 0.1]

    sol_b1 = solve_ode(pred_prey_eq, [0.5, 0.5], t_eval, deltat_max, rk4_step, 1, vars1)
    sol_b2 = solve_ode(pred_prey_eq, [0.5, 0.5], t_eval, deltat_max, rk4_step, 1, vars2)

    plt.subplot(2, 1, 1)
    plt.title('b = ' + str(b1))
    plt.xlabel('t'), plt.ylabel('x and y')
    plt.plot(t_eval, sol_b1[1], 'r', label='y'), plt.plot(t_eval, sol_b1[0], 'g', label='x')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.title('b = '+str(b2))
    plt.xlabel('t'), plt.ylabel('x and y')
    plt.plot(t_eval, sol_b2[1], 'r', label='y'), plt.plot(t_eval, sol_b2[0], 'g', label='x')
    plt.legend()

    plt.tight_layout(pad=1)
    plt.show()


def x_period(x, t_eval, dp=4):
    """
    This function finds the time period of the x values
    :param x: List of x values
    :param t_eval: The t values corresponding to the values of x
    :param dp: The number of decimal places to round all the values of x to
    :return: returns the most common value in x and the Time period of x
    """

    round_x = [np.round(num, dp) for num in x]  # round all values in list to dp decimal places

    most_common_x = Counter(round_x).most_common(1)[0][0]

    # find the time values of the most common value
    all_t_val = t_eval[np.where(round_x == most_common_x)]

    # calculate the time period by getting the time period between these values
    t_val_list = []
    for i in range(len(all_t_val) - 1):
        t_val_list.append(all_t_val[i + 1] - all_t_val[i])

    t_per = min(t_val_list)

    return most_common_x, t_per


def main():
    t_eval, deltat_max, vars1 = np.linspace(0, 1000, 1000), 0.01, [1, 0.1, 0.1]

    sol_pred_prey = solve_ode(pred_prey_eq, [0.5, 0.5], t_eval, deltat_max, rk4_step, 1, vars1)
    # one value > 0.26 and one value < 0.26 are chosen to observe how the behaviour changes either side of 0.26
    # compare_b_values(0.1, 0.5)
    most_common_x, T = x_period(sol_pred_prey[0], t_eval)
    print(T)


if __name__ == '__main__':
    main()
