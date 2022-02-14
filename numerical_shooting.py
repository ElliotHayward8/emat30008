import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
from ODE_Solvers import euler_step, rk4_step, solve_ode, solve_to


def pred_prey_eq(X, t, a=1, b=0.26, d=0.1):
    x = X[0]
    y = X[1]
    dxdt = x * (1 - x) - (a * x * y) / (d + x)
    dydt = b * y * (1 - (y / x))
    u = [dxdt, dydt]
    return u


t_eval = np.linspace(0, 100)
deltat_max = 0.0001

# sol = solve_ode(pred_prey_eq, [0.5, 0.5], t_eval, deltat_max, rk4_step, 1)
sol = odeint(pred_prey_eq, [1, 1], t_eval)

plt.plot(t_eval, sol[:, 1], 'r')
plt.plot(t_eval, sol[:, 0], 'g')
plt.show()
