import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
from collections import Counter
from ODE_Solvers import euler_step, rk4_step, solve_ode, solve_to
from value_checks import ode_checker, array_int_or_float


# predator prey equation
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
    a, b, d = pars[0][0], pars[0][1], pars[0][2]
    dxdt = x * (1 - x) - (a * x * y) / (d + x)
    dydt = b * y * (1 - (y / x))
    return np.array([dxdt, dydt])


# define the phase condition for the predator prey equations
def pred_prey_phase_cond(x0, pars):
    return pred_prey_eq(x0, 0, pars)[0]


def compare_b_values(b1, b2):
    """
    A function which produces a graph to compare how the predator prey model changes for different values of b using
    the rk4 method with initial conditions: x,y = 0.5 and h = 0.001
    :param b1: First value of b
    :param b2: Second value of b
    """

    t_eval = np.linspace(0, 100, 100)
    deltat_max = 0.001

    pars1, pars2 = [1, b1, 0.1], [1, b2, 0.1]

    sol_b1 = solve_ode(pred_prey_eq, [0.5, 0.5], t_eval, deltat_max, 'rk4', 1, pars1)
    sol_b2 = solve_ode(pred_prey_eq, [0.5, 0.5], t_eval, deltat_max, 'rk4', 1, pars2)

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


def shooting(f):
    """
    Construct the shooting root-finding problem for a given ODE
    :param f: ODE to use the shooting root-finding method on
    :return: Returns the function G which can be solved to find the root which will solve the shooting problem
    """
    # print('start of shooting ' + str(pars))

    def G(u0T, phase_con, *pars):
        """
        Function which should have a root which returns the periodic orbit of an ODE/ system of ODEs
        :param u0T: Array which contains the starting guess of the coordinates and the time period
        :param phase_con: Function of the phase condition
        :param pars: Array of any additional parameters
        :return:
        """
        # print('start of G ' + str(pars))

        def F(u0, T):
            """
            Solution of ODE f at T with initial conditions x0 using the rk4 method
            :param u0: Initial condition(s) for the ODE
            :param T: The time value to solve at
            :return: Returns the solution of the ODE at time T using the rk4 method
            """
            t_eval = np.linspace(0, T, 1000)
            # print('F ' + str(pars))
            sol = solve_ode(f, u0, t_eval, 0.01, 'rk4', True, *pars)
            return sol[:, -1]

        T, u0 = u0T[-1], u0T[:-1]

        # construct an array of the initial guess minus the solution alongside the phase condition
        g = np.append(u0 - F(u0, T), phase_con(u0, *pars))
        return g
    return G


def find_shooting_orbit(f, u0T, phase_cond, *pars):
    """
    Function which finds the starting coordinates and time period of a periodic orbit within an ODE
    :param f: An ODE to find the time period and orbit coordinates for
    :param u0T: Array of the initial guess of the orbit location
    :param phase_cond: Phase condition for the shooting problem
    :param pars: Array of any additional parameters
    :return: Returns the starting coordinates and time period of the ODE
    """
    # Check the values of u0T
    array_int_or_float(u0T, 'u0T')

    # Check the inputted ODE is formatted correctly
    ode_checker(f, u0T[:-1], [u0T[-1]], *pars)

    # Check the inputted phase condition is formatted correctly
    if callable(phase_cond):

        pc_val = phase_cond(u0T[:-1], *pars)

        # check the phase condition returns an int or float
        if not isinstance(pc_val, (int, float, np.int_, np.float_)):
            raise TypeError(f"Output of f is of the type {type(pc_val)}. It should be an int or a float")

    else:
        raise TypeError(f"phase_cond: '{phase_cond}' must be a callable function.")

    G = shooting(f)
    shooting_orbit = fsolve(G, u0T, args=(phase_cond, *pars))
    return shooting_orbit


def main():
    t_eval, deltat_max, pars1 = np.linspace(0, 1000, 1000), 0.01, [1, 0.2, 0.1]

    pred_prey_u0T = np.array([0.58, 0.285, 21])

    sol_pred_prey = solve_ode(pred_prey_eq, pred_prey_u0T[:-1], t_eval, deltat_max, 'rk4', 1, pars1)

    # one value > 0.26 and one value < 0.26 are chosen to observe how the behaviour changes either side of 0.26
    # compare_b_values(0.1, 0.5)

    shooting_orbit = find_shooting_orbit(pred_prey_eq, pred_prey_u0T, pred_prey_phase_cond, pars1)

    print(shooting_orbit)

    #plt.plot(shooting_orbit[0], shooting_orbit[1], 'go', label='Shooting Orbit')
    #plt.plot(sol_pred_prey[0], sol_pred_prey[1], 'b', label='Solution')
    #plt.xlabel('x'), plt.ylabel('y'), plt.legend()

    plt.plot(t_eval, sol_pred_prey[0], 'r', label='x')
    plt.plot(t_eval,  sol_pred_prey[1], 'g', label='y')
    plt.show()


if __name__ == '__main__':
    main()
