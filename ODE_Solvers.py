import matplotlib.pyplot as plt
import numpy as np


def f(x, t): # This function defines the differential equation
    return x


def euler_step(f, x, t, h): # Perform one euler step
    x += (h * f(x, t))
    t += h
    return x, t


def rk4_step(f, x, t, h): # Perform one step of the Runge-Kutta method
    half_h = h / 2
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


N = 10
x_error_list, deltat_max_list = [], []
x0, t0, t1 = 0, 0, 1
for deltat_max in np.logspace(-15, -1, N):
    x1, t1 = solve_to(f, x0, t0, t1, deltat_max, euler_step)
    print(deltat_max)
    x_error_list.append(np.exp(1) - x1)
    deltat_max_list.append(deltat_max)

print(x_error_list)
print(deltat_max_list)
plt.loglog(x_error_list, deltat_max_list)
plt.show()


# x1, t1 = solve_to(1, 0, 1, 0.25, euler_step)
# print(x1, t1)
# x2RK, t2RK = solve_to(1, 0, 1, 1, rk4_step)
# print(x2RK, t2RK)
# t_eval = [0, 1, 2, 3, 4, 5]
# x_list, t_eval = solve_ode(f, 1, t_eval, 0.0001, rk4_step)
# print(x_list)
# print(t_eval)
