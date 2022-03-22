import matplotlib.pyplot as plt
import numpy as np
import time
from value_checks import array_int_or_float, ode_checker


def func1(x, t):
    """
    This function defines the ODE dx/dt = x
    :param x: Value of x
    :param t: Time value
    :return: Value of dx/dt = x
    """
    return x


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


def euler_step(f, x0, t0, h, *pars):
    """
    This function performs one euler step
    :param f: Function defining an ODE or system of ODEs
    :param x0: Starting x value(s)
    :param t0: Starting time value
    :param h: Designated step size
    :param pars: Array of any additional parameters
    :return: returns the value of function after 1 step (at t1) and t1
    """
    x1 = x0 + h * f(x0, t0, *pars)
    t1 = t0 + h
    return x1, t1


def rk4_step(f, x0, t0, h, *pars):
    """
    This function performs one step of the RK4 method
    :param f: Function defining an ODE or system of ODEs
    :param x0: Starting x value(s)
    :param t0: Starting time value
    :param h: Designated step size
    :param pars: Array of any additional parameters
    :return: returns the value of function after 1 step (at t1) and the value t1
    """
    half_h = h / 2
    k1 = f(x0, t0, *pars)
    k2 = f(x0 + (half_h * k1), t0 + half_h, *pars)
    k3 = f(x0 + (half_h * k2), t0 + half_h, *pars)
    k4 = f(x0 + (h * k3), t0 + h, *pars)
    x1 = x0 + ((h / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
    t1 = t0 + h
    return x1, t1


def solve_to(f, x0, t0, t1, deltat_max, solver, *pars):
    """
    Solves the ODE (f) between t0 and t1 with IC = x0
    :param f: Function defining an ODE or system of ODEs
    :param x0: Starting x value(s)
    :param t0: Starting time value
    :param t1: Final time value
    :param deltat_max: Maximum step size (maximum value of h)
    :param solver: Defines which solver to use ('euler'/'rk4')
    :param pars: Array of any additional parameters
    :return: X value at time t1
    """
    h = deltat_max
    t, x = t0, x0

    while t < t1:
        if t + deltat_max > t1:
            h = t1 - t
        if solver == 'euler':
            x, t = euler_step(f, x, t, h, *pars)
        elif solver == 'rk4':
            x, t = rk4_step(f, x, t, h, *pars)
        else:
            raise NameError(f"solver : {solver} isn't present (must select 'euler' or 'rk4')")
    return x


def solve_ode(f, x0, t_eval, deltat_max, solver, ODEs, *pars):
    """
    Calculates the value of an ODE or system of ODEs over a range of time values
    :param f: Function defining an ODE or system of ODEs
    :param x0: Starting x value(s)
    :param t_eval: Array of time values to solve at
    :param deltat_max: Maximum step size (maximum value of h)
    :param solver: Which solver to use ('euler'/'rk4')
    :param ODEs: True/False defining whether it is a system of ODEs or not
    :param pars: Array of any additional parameters
    :return: Returns an array of x values at each time value in t_eval
    """
    # print('start solve_ode ' + str(pars))
    # Check the values of x0, t_eval and deltat_max
    array_int_or_float(x0, 'x0 (initial condition)')
    array_int_or_float(t_eval, 't_eval')
    array_int_or_float(deltat_max, 'deltat_max')

    # Check the inputted ODE is formatted correctly
    ode_checker(f, x0, t_eval, *pars)

    # define the empty x array depending on size of x0 and t_eval
    if ODEs:
        x_array = np.empty(shape=(len(t_eval), len(x0)))
    else:  # if we have a first oder ODE len() doesn't work
        x_array = np.empty(shape=(len(t_eval), 1))
    x_array[0] = x0

    for n in range(len(t_eval) - 1):
        xn = solve_to(f, x_array[n], t_eval[n], t_eval[n+1], deltat_max, solver, *pars)
        x_array[n + 1] = xn

    if ODEs:
        x_array = x_array.transpose()

    return x_array


def func1_error_graph(f, N, x0, t0, t1, *pars):
    """
    This function creates an error graph for the two methods for the function dx/dt = x
    :param f: Function defining an ODE or system of ODEs
    :param N: The lowest order of h to calculate the error of (if N = 5, h = 10^-5)
    :param x0: Starting x value(s)
    :param t0: Starting time value
    :param t1: Final time value
    :param pars: Array of any additional parameters
    """
    x_error_list, deltat_max_list, xn_error_list = [], [], []
    for deltat_max in np.logspace(-N, -1, 2*N):
        x1 = solve_to(f, x0, t0, t1, deltat_max, 'euler', *pars)
        xn = solve_to(f, x0, t0, t1, deltat_max, 'rk4', *pars)
        x_error_list.append(abs(np.exp(1) - x1))
        xn_error_list.append(abs(np.exp(1) - xn))
        deltat_max_list.append(deltat_max)

    plt.loglog(deltat_max_list, x_error_list, label='Euler Method')
    plt.loglog(deltat_max_list, xn_error_list, 'r-', label='RK4 Method')
    plt.ylabel('|$x_{n}- x(t_{n})$|')
    plt.xlabel('h')
    plt.title('Order of error for Euler and RK4')
    plt.legend()
    plt.show()


def time_methods(f, x0, t0, t1, *pars):
    """
    This function times the two methods when they have the same error over 10,000 runs (For the report I used f = func1,
    x0 = 1, t0 = 0 and t1 = 1
    :param f: Function defining an ODE or system of ODEs
    :param x0: Starting x value(s)
    :param t0: Starting time value
    :param t1: Final time value
    :param pars: Array of any additional parameters
    """
    time0 = time.time()
    n = 0
    while n < 10000:
        x1 = solve_to(f, x0, t0, t1, 0.00068895, 'euler', *pars)
        n += 1
    time1 = time.time()
    n = 0
    while n < 10000:
        xn = solve_to(f, x0, t0, t1, 0.5, 'rk4', *pars)
        n += 1
    time2 = time.time()
    print('Euler time = ' + str(time1 - time0))
    print('RK4 time = ' + str(time2 - time1))


def true_func2(t):
    """
    Function which calculates the true values of the results of the ODEs dx/dt = y and dy/dt = -x
    :param t: Time value
    :return: Returns an array of x and y at time t
    """
    x = np.sin(t) + np.cos(t)
    y = np.cos(t) - np.sin(t)
    return np.array([x, y])


def func2_comparison_graph(deltat_max, time_periods, x0, total_time, *pars):
    """
    This function produces graphs of x and y to compare how the RK4 and Euler method perform compared to the actual
    solution of the system of ODEs dx/dt = y and dy/dt = -x
    :param deltat_max: Maximum value of h
    :param time_periods: Number of time periods
    :param x0: Initial value(s) of x
    :param total_time: time to run models over
    :param pars: Array of any additional parameters
    """
    t = np.linspace(0, total_time, time_periods)
    euler_sol = solve_ode(func2, x0, t, deltat_max, 'euler', True, *pars)
    euler_sol_x, euler_sol_y = euler_sol[0], euler_sol[1]
    rk4_sol = solve_ode(func2, x0, t, deltat_max, 'rk4', True, *pars)
    rk4_sol_x, rk4_sol_y = rk4_sol[0], rk4_sol[1]
    sol = true_func2(t)
    sol_x, sol_y = sol[0], sol[1]

    plt.plot(euler_sol_x, euler_sol_y, label='Euler method')
    plt.plot(rk4_sol_x, rk4_sol_y, label='RK4 method')
    plt.plot(sol_x, sol_y, 'r', label='True value')
    plt.legend(), plt.xlabel('x'), plt.ylabel('y')
    plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(t, euler_sol_y, label='Euler method')
    plt.plot(t, rk4_sol_y, label='RK4 method')
    plt.plot(t, sol_y, label='True value')
    plt.legend(), plt.xlabel('t'), plt.ylabel('y')

    plt.subplot(2, 1, 2)
    plt.plot(t, euler_sol_x, label='Euler method')
    plt.plot(t, rk4_sol_x, label='RK4 method')
    plt.plot(t, sol_x, label='True value')
    plt.legend(), plt.xlabel('t'), plt.ylabel('x')
    plt.show()


def main():
    func1_error_graph(func1, 5, 1, 0, 1)

    time_methods(func1, 1, 0, 1)

    func2_comparison_graph(0.1, 1000, [1, 1], 50)


if __name__ == '__main__':
    main()
