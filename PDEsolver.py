# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from value_checks import array_int_or_float


# Solve the PDE: loop over all the time points
def forward_euler(u_i_func, mx, mt, kappa, L, T, bc_0, bc_L):
    """
    A function which performs the forward euler methpd on the heat equation
    :param u_i_func: Function which defines the prescribed initial temperature
    :param mx: Number of grid points in space
    :param mt: Number of grid points in time
    :param kappa: Diffusion constant
    :param L: Length of the spacial domain
    :param T: Total time to solve for
    :param bc_0: Boundary condition at x = 0
    :param bc_L: Boundary condition at x = L
    :return: Solution of PDE at time T
    """

    # Check that the boundary conditions given are an integer or a float
    if not isinstance(bc_0, (int, np.int_, float, np.float_)):
        raise TypeError(f"bc_0: {bc_0} is not an integer")

    if not isinstance(bc_L, (int, np.int_, float, np.float_)):
        raise TypeError(f"bc_0: {bc_L} is not an integer")

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx + 1)  # mesh points in space
    t = np.linspace(0, T, mt + 1)  # mesh points in time
    deltax = x[1] - x[0]  # grid spacing in x
    deltat = t[1] - t[0]  # grid spacing in t

    # calculate the value of lambda, stability requires 0 < lambda < 0.5
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number

    # print the value of variables
    print("deltax = ", deltax), print("deltat = ", deltat), print("lambda = ", lmbda)

    u_j, u_jp1 = np.zeros(x.size), np.zeros(x.size)  # u at current and next time step

    # Set initial condition
    for i in range(0, mx + 1):
        u_j[i] = u_i_func(x[i])

    for j in range(0, mt):
        # Forward Euler time step at inner mesh points
        # PDE discretised at position x[i], time t[j]
        for i in range(1, mx):
            u_jp1[i] = u_j[i] + lmbda*(u_j[i-1] - 2*u_j[i] + u_j[i+1])
        
        # Boundary conditions
        u_jp1[0], u_jp1[mx] = bc_0, bc_L
        
        # Save u_j at time t[j+1]
        u_j[:] = u_jp1[:]

    return x, u_j


def fe_matrix_vector_form(u_i_func, mx, mt, kappa, L, T, bc_0, bc_L):
    """
    A function which performs the forward Euler schem in matrix/vector form on the heat equation
    :param u_i_func: Function which defines the prescribed initial temperature
    :param mx: Number of grid points in space
    :param mt: Number of grid points in time
    :param kappa: Diffusion constant
    :param L: Length of the spacial domain
    :param T: Total time to solve for
    :param bc_0: Boundary condition at x = 0
    :param bc_L: Boundary condition at x = L
    :return: Solution of PDE at time T
    """

    # Check that the boundary conditions given are an integer or a float
    if not isinstance(bc_0, (int, np.int_, float, np.float_)):
        raise TypeError(f"bc_0: {bc_0} is not an integer or float")

    if not isinstance(bc_L, (int, np.int_, float, np.float_)):
        raise TypeError(f"bc_0: {bc_L} is not an integer or float")

    # Set up the numerical environment variables
    x, t = np.linspace(0, L, mx + 1), np.linspace(0, T, mt + 1)    # mesh points in space and time
    deltax, deltat = x[1] - x[0], t[1] - t[0]  # grid spacing in x and t

    # calculate the value of lambda, stability requires 0 < lambda < 0.5
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number
    lmbda_vec, main_diag_vec = [lmbda] * (mx - 2), [1 - (2 * lmbda)] * (mx - 1)

    # create the A_FE tridiagonal matrix
    main_diag_matrix = np.diag(main_diag_vec, 0)
    up_matrix, low_matrix = np.diag(lmbda_vec, 1), np.diag(lmbda_vec, -1)

    a_fe = main_diag_matrix + up_matrix + low_matrix

    u_j, u_jp1 = np.zeros(x.size), np.zeros(x.size)  # u at current and next time step

    # Set initial condition
    for i in range(0, mx + 1):
        u_j[i] = u_i_func(x[i])

    sliced_u_j = np.vstack(u_j[1:-1])

    u_jp1 = a_fe.dot(sliced_u_j)

    u_jp1 = np.append(u_jp1, bc_L)
    u_j = np.insert(u_jp1, 0, bc_0, axis=0)

    for j in range(0, mt):
        # Forward Euler time step at inner mesh points
        # PDE discretised at position x[i], time t[j]
        sliced_u_j = np.vstack(u_j[1:-1])

        u_j = a_fe.dot(sliced_u_j)
        # Boundary conditions
        u_j = np.append(u_j, bc_L)
        u_j = np.insert(u_j, 0, bc_0, axis=0)

    return x, u_j


def main():
    # Set problem parameters/functions
    kappa = 1.0  # diffusion constant - how easily diffusion occurs
    L = 1.0         # length of spatial domain
    T = 100        # total time to solve for

    def u_i(x, p=1):
        # initial temperature distribution
        y = (np.sin(pi * x / L)) ** p
        return y

    def u_exact(x, t):
        # the exact solution
        y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
        return y

    x, u_j = forward_euler(u_i, 10, 100000, kappa, L, T, 0, 0)

    z_fe, u_j_fe = fe_matrix_vector_form(u_i, 10, 100000, kappa, L, T, 0, 0)


    xx = np.linspace(0, L, 250)

    # Plot the final result and exact solution
    plt.plot(x, u_j, 'ro', label='num')
    plt.plot(xx, u_exact(xx, T), 'b-', label='exact')
    plt.xlabel('x')
    plt.ylabel('u(x,' + str(T) + ')')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
