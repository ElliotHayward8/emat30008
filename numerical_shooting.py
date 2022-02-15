import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
from ODE_Solvers import euler_step, rk4_step, solve_ode, solve_to


def pred_prey_eq(X, t, *vars):
    x = X[0]
    y = X[1]
    a, b, d = vars[0][0], vars[0][1], vars[0][2]
    dxdt = x * (1 - x) - (a * x * y) / (d + x)
    dydt = b * y * (1 - (y / x))
    return np.array([dxdt, dydt])


t_eval = np.linspace(0, 1000, 10000)
deltat_max = 0.001
vars = [1, 0.15, 0.1]

sol = solve_ode(pred_prey_eq, [0.5, 0.5], t_eval, deltat_max, rk4_step, 1, vars)

plt.plot(sol[0], sol[1], 'r')
plt.ylabel('y')
plt.xlabel('x')
plt.show()

plt.plot(t_eval, sol[0], 'r')
plt.xlabel('t')
plt.ylabel('x')
plt.show()
