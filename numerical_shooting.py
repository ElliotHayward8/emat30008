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


def inspection_xy_period(x, y, t_eval, dp=4):
    """
    This function finds the time period of the x values
    :param x: List of x values
    :param y: List of y values
    :param t_eval: The t values corresponding to the values of x
    :param dp: The number of decimal places to round all the values of x to
    :return: returns the starting x and y value of an orbit alongside the time period of the orbit
    """

    round_x = [np.round(num, dp) for num in x]  # round all values in list to dp decimal places
    most_common_x = Counter(round_x).most_common(1)[0][0] # find the most common x value

    # find the time values of the most common value
    all_t_val = t_eval[np.where(round_x == most_common_x)]

    # calculate the time period by getting the time separation between these values
    t_val_list = []
    for i in range(len(all_t_val) - 1):
        t_val_list.append(all_t_val[i + 1] - all_t_val[i])

    # Take correct time period as the smallest value within the list
    t_per = min(t_val_list)

    start_index = round_x.index(most_common_x)
    most_common_y = y[start_index]

    return most_common_x, most_common_y, t_per


def shooting(f):
    """
    Construct the shooting root-finding problem for a given ODE
    :param f: ODE to use the shooting root-finding method on
    :return: Returns the function G which can be solved to find the root which will solve the shooting problem
    """
    def G(u0T, phase_con, *vars):
        """
        Function which should have a root which returns the periodic orbit of an ODE/ system of ODEs
        :param u0T: Array which contains the starting guess of the coordinates and the time period
        :param phase_con: Function of the phase condition
        :param vars: List of additional variables
        :return:
        """

        def F(u0, T):
            """
            Solution of ODE f at T with initial conditions x0 using the rk4 method
            :param u0: Initial condtion(s) for the ODE
            :param T: The time value to solve at
            :return: Returns the solution of the ODE at time T using the rk4 method
            """
            t_eval = np.linspace(0, T, 1000)

            sol = solve_ode(f, u0, t_eval, 0.01, rk4_step, True, *vars)
            return sol[:, -1]

        T, u0 = u0T[-1], u0T[:-1]

        # construct an array of the initial guess minus the solution alongside the phase condition
        g = np.append(u0 - F(u0, T), phase_con(u0, *vars))
        return g
    return G


def find_shooting_orbit(f, u0T, phase_cond, *vars):
    G = shooting(f)
    shooting_orbit = fsolve(G, u0T, args=(phase_cond, *vars))
    return shooting_orbit


# define the phase condition for the predator prey equations
def pred_prey_phase_cond(x0, vars):
    return pred_prey_eq(x0, 0, vars)[0]


def main():
    t_eval, deltat_max, vars1 = np.linspace(0, 1000, 1000), 0.01, [1, 0.1, 0.1]

    pred_prey_u0T = np.array([0.5, 0.5, 23])

    sol_pred_prey = solve_ode(pred_prey_eq, pred_prey_u0T[:-1], t_eval, deltat_max, rk4_step, 1, vars1)

    # one value > 0.26 and one value < 0.26 are chosen to observe how the behaviour changes either side of 0.26
    # compare_b_values(0.1, 0.5)
    start_x, start_y, T = inspection_xy_period(sol_pred_prey[0], sol_pred_prey[1], t_eval)

    shooting_orbit = find_shooting_orbit(pred_prey_eq, pred_prey_u0T, pred_prey_phase_cond, vars1)

    print(shooting_orbit)
    print(np.array([start_x, start_y, T]))




if __name__ == '__main__':
    main()
