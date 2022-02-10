import matplotlib.pyplot as plt
import numpy as np
import time

def func1(x, t): # This function defines the differential equation
    return x


def func2(x, t):  # This function defines a second differential equation
    x[0] = x[1]
    x[1] = -x[0]
    return x


def euler_step(f, x, t, h): # Perform one euler step
    if type(x) is list:
        fx = f(x,t)
        for element in fx:
            fx.append(element)
        x = [a + b for a,b in zip(x, fx)]
        print(x)
    else:
        x += (h * f(x, t))
    t += h
    return x, t


def rk4_step(f, x, t, h): # Perform one step of the Runge-Kutta method
    half_h = h / 2
    if type(x) is list:
        x = 1
    else:
        k1 = f(x, t)
        k2 = f(x + (half_h * k1), t + half_h)
        k3 = f(x + (half_h * k2), t + half_h)
        k4 = f(x + (h * k3), t + h)
        x += h * 1/6 * (k1 + 2*k2 + 2*k3 + k4)
    t += h
    return x, t

def solve_to(f, x0, t0, t1, deltat_max, solver): # solve between two t values with an initial condition x1
    h = deltat_max
    t, x = t0, x0
    while t < t1:
        if t + deltat_max > t1:
            h = t1 - t
        x, t = solver(f, x, t, h)
    return x, t1


def solve_ode(f, x0, t_eval, deltat_max, solver):
    n, x = 0, x0
    x_list = []
    x_list.append(x0)
    while n < len(t_eval) - 1:
        x, t1 = solve_to(f, x, t_eval[n], t_eval[n+1], deltat_max, solver)
        x_list.append(x)

        n += 1
    return x_list, t_eval


def func1_error_graph(f, solve_to, N, x0, t0, t1):  # Create values for an error graph of RK4/Euler method as h changes
    x_error_list, deltat_max_list, xn_error_list = [], [], []
    for deltat_max in np.logspace(-N, -0.5, 2*N):
        x1, t1 = solve_to(f, x0, t0, t1, deltat_max, euler_step)
        xn, tn = solve_to(f, x0, t0, t1, deltat_max, rk4_step)
        x_error_list.append(abs(np.exp(1) - x1))
        xn_error_list.append(abs(np.exp(1) - xn))
        deltat_max_list.append(deltat_max)
    return x_error_list, xn_error_list, deltat_max_list


x, t = euler_step(func2, [0.5,-0.5], 0, 1)

# x, t1 = solve_to(func2, [0.5,-0.5], 0, 0.86, 0.0001, euler_step)
# print(x)



# time0 = time.time()
# n = 0
# while n < 10000:
#     x1, t1 = solve_to(f, 1, 0, 1, 0.00068895, euler_step)
#     n += 1
# time1 = time.time()
# n = 0
# while n < 10000:
#     xn, tn = solve_to(f, 1, 0, 1, 0.5, rk4_step)
#     n += 1
# time2 = time.time()
# print('Euler time = ' + str(time1 - time0))
# print('RK4 time = ' + str(time2 - time1))
'''
I used the code above to time the 2 methods over 10,000 runs
I used the lines of code below to generate an error graph for the 2 methods
'''
# x_error_list, xn_error_list, deltat_max_list = func1_error_graph(func1, solve_to, 5, 1, 0, 1)
# plt.loglog(deltat_max_list, x_error_list, label='Euler Method')
# plt.loglog(deltat_max_list, xn_error_list, 'r-', label='RK4 Method')
# plt.ylabel('|$x_{n}- x(t_{n})$|')
# plt.xlabel('h')
# plt.title('Order of error for Euler and RK4')
# plt.legend()
# plt.show()

