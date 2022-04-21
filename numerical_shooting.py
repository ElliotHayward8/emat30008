import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
from collections import Counter
from ODE_Solvers import euler_step, rk4_step, solve_ode, solve_to
from value_checks import ode_checker, array_int_or_float


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


# Define the phase condition for the predator prey equations
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
    Construct the function which returns the values to be set to 0 for the shooting root-finding problem
    :param f: ODE to use the shooting root-finding method on
    :return: Returns the function G which can be solved to find the root which solves the shooting problem
    """

    def G(u0T, phase_con, *pars):
        """
        Function which should have a root which returns the periodic orbit of an ODE/ system of ODEs
        :param u0T: Array which contains the starting guess of the coordinates and the time period
        :param phase_con: Function for the phase condition of the problem
        :param pars: Array of any additional parameters
        :return: Returns an array of the value of the phase condition and the difference between the
        """

        def F(u0, T):
            """
            Solution of ODE (f) at time T with initial conditions x0 using the rk4 method from ODE_Solvers.py
            :param u0: Initial condition(s) for the ODE
            :param T: The time value to solve at
            :return: Returns the solution of the ODE at time T using the rk4 method
            """

            # Create a list of time values to solve over
            t_eval = np.linspace(0, T, 1000)
            sol = solve_ode(f, u0, t_eval, 0.01, 'rk4', True, *pars)

            # Extract the final solution
            final_sol = sol[:, -1]

            # return the final solution value
            return final_sol

        # Extract the inputted time and initial values
        T, u0 = u0T[-1], u0T[:-1]

        """
        return an array of the initial guess minus the solution alongside the phase condition this is then to be 
        minimised using a solver to find where these values are 0 in order to isolate a periodic orbit
        """

        return np.append(u0 - F(u0, T), phase_con(u0, *pars))
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

        # Check the phase condition returns an int or float
        if not isinstance(pc_val, (int, float, np.int_, np.float_)):
            raise TypeError(f'Output of f is of the type {type(pc_val)}. It should be an int or a float')

    else:
        raise TypeError(f'phase_cond: \'{phase_cond}\' must be a callable function.')

    G = shooting(f)
    fsolve_sol = fsolve(G, u0T, (phase_cond, *pars), full_output=True)
    shooting_orbit = fsolve_sol[0]
    converge = fsolve_sol[2]

    if converge == 1:
        return shooting_orbit
    else:
        raise ValueError("fsolve was unable to converge")


def plot_isolated_orbit(f, shooting_orbit, ODEs, *pars):
    """
    Function which finds the starting coordinates and time period of a periodic orbit within an ODE
    :param f: ODE that the shooting function has been performed on
    :param shooting_orbit: Initial condition and time period found
    :param ODEs: A boolean variable which defines whether it is a singular or system of ODE(s) (0 = singular)
    :param pars: Array of any additional parameters
    """
    u0, T, deltatmax = shooting_orbit[:-1], shooting_orbit[-1], 0.01
    t_eval = np.linspace(0, T, int(T * 2))

    iso_orbit = solve_ode(f, u0, t_eval, deltatmax, 'rk4', ODEs, *pars)

    if len(iso_orbit) < 5:
        if len(iso_orbit) == 1:
            plt.plot(t_eval, iso_orbit[0], 'r-', label='x')
        elif len(iso_orbit) == 2:
            plt.plot(t_eval, iso_orbit[0], 'r-', label='x')
            plt.plot(t_eval, iso_orbit[1], 'g-', label='y')
        elif len(iso_orbit) == 3:
            plt.plot(t_eval, iso_orbit[0], 'r-', label='x')
            plt.plot(t_eval, iso_orbit[1], 'g-', label='y')
            plt.plot(t_eval, iso_orbit[2], 'b-', label='z')
        elif len(iso_orbit) == 4:
            plt.plot(t_eval, iso_orbit[0], 'r-', label='x')
            plt.plot(t_eval, iso_orbit[1], 'g-', label='y')
            plt.plot(t_eval, iso_orbit[2], 'b-', label='z')
            plt.plot(t_eval, iso_orbit[3], 'y-', label='w')
    else:
        raise TypeError('This function only plots orbits for ODEs with 4 or less variables')

    plt.title('Isolated Periodic Orbit')
    plt.legend(), plt.xlabel('Time'), plt.ylabel('Value of variables')
    plt.show()


def main():
    # Define the required values to solve the ODE using solve_ode
    t_eval, deltat_max, pars1 = np.linspace(0, 50, 100), 0.01, [1, 0.2, 0.1]

    """
    Define a close initial guess of the solution of the shooting problem for the predator-prey equations in the form 
    (x,y,T) where T is the time period of an orbit
    """
    pred_prey_u0T = np.array([0.58, 0.285, 21])

    sol_pred_prey = solve_ode(pred_prey_eq, pred_prey_u0T[:-1], t_eval, deltat_max, 'rk4', 1, pars1)

    """
    Compare how the predator prey equations change depending on how the value of b changes, when a = 1 and d = 0.1,
    here the behaviour is compared when b = 0.1 and b = 0.5 so that the change in behaviour either side of b = 0.26 can 
    be observed
    """

    compare_b_values(0.1, 0.5)

    """
    Find the solution to the shooting problem of the predator-prey equations and plot the isolated orbit
    """

    shooting_orbit = find_shooting_orbit(pred_prey_eq, pred_prey_u0T, pred_prey_phase_cond, pars1)

    print('The time period of the isolated periodic orbit is : ' + str(shooting_orbit[-1]))

    plot_isolated_orbit(pred_prey_eq, shooting_orbit, 1, pars1)

    """
    Plot the x and y values of the predator-prey equations against time 
    """
    plt.title('Solution of the Predator-Prey Equations')
    plt.plot(t_eval, sol_pred_prey[0], 'r', label='x')
    plt.plot(t_eval,  sol_pred_prey[1], 'g', label='y')
    plt.xlabel('Time'), plt.ylabel('Value of variables')
    plt.show()


if __name__ == '__main__':
    main()
