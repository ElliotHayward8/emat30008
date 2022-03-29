# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


# Use this function, which utilises the fact it is a sparse matrix, as it reduces the storage requirements
# It also speeds up any calculations
def create_tri_diag_mat(main_diag_len, low_diag, main_diag, high_diag):
    """
    Function which creates a tridiagonal matrix
    :param main_diag_len: Length of the main diagonal
    :param low_diag: Value for the lower diagonal
    :param main_diag: Value for the main diagonal
    :param high_diag: Value for the upper diagonal
    :return: Tridiagonal matrix with designated values
    """
    low_diags = [low_diag] * (main_diag_len - 1)
    high_diags = [high_diag] * (main_diag_len - 1)
    main_diags = [main_diag] * main_diag_len

    diagonals = [low_diags, main_diags, high_diags]
    tri_diag_mat = diags(diagonals, [-1, 0, 1], format='csr')
    return tri_diag_mat


# Solve the PDE: loop over all the time points
def forward_euler(u_i_func, mx, mt, kappa, L, T, bc_0, bc_L):
    """
    A function which performs the forward euler method on the heat equation
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

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx + 1)  # mesh points in space
    t = np.linspace(0, T, mt + 1)  # mesh points in time
    deltax = x[1] - x[0]  # grid spacing in x
    deltat = t[1] - t[0]  # grid spacing in t

    # calculate the value of lambda, stability requires 0 < lambda < 0.5
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number

    # Check lambda is within the stable range
    if lmbda <= 0 or lmbda >= 0.5:
        raise ValueError(f"lmbda: {lmbda} is not within the range 0 < lmbda < 0.5")

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


def be_matrix_vector_form(u_i_func, mx, mt, kappa, L, T, bc_0, bc_L):
    """
    A function which performs the backward Euler scheme in matrix/vector form on the heat equation
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

    # Set up the numerical environment variables
    x, t = np.linspace(0, L, mx + 1), np.linspace(0, T, mt + 1)  # mesh points in space and time
    deltax, deltat = x[1] - x[0], t[1] - t[0]  # grid spacing in x and t

    # calculate the value of lambda
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number

    # create the A_BE tridiagonal matrix
    a_be = create_tri_diag_mat(mx - 1, -lmbda, 1 + (2 * lmbda), -lmbda)
    u_j, u_jp1 = np.zeros(x.size), np.zeros(x.size)  # u at current and next time step

    # Set initial condition
    for i in range(0, mx + 1):
        u_j[i] = u_i_func(x[i])

    sliced_u_j = np.vstack(u_j[1:-1])

    u_jp1 = spsolve(a_be, sliced_u_j)

    u_jp1 = np.append(u_jp1, bc_L)
    u_j = np.insert(u_jp1, 0, bc_0, axis=0)

    for j in range(0, mt - 1):
        # Forward Euler time step at inner mesh points
        # PDE discretised at position x[i], time t[j]
        sliced_u_j = np.vstack(u_j[1:-1])

        u_j = spsolve(a_be, sliced_u_j)
        # Boundary conditions
        u_j = np.append(u_j, bc_L)
        u_j = np.insert(u_j, 0, bc_0, axis=0)

    return x, u_j


def c_n(u_i_func, mx, mt, kappa, L, T, bc_0, bc_L):
    """
    A function which performs the Crank Nicholson scheme in matrix/vector form on the heat equation
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
    # Set up the numerical environment variables
    x, t = np.linspace(0, L, mx + 1), np.linspace(0, T, mt + 1)  # mesh points in space and time
    deltax, deltat = x[1] - x[0], t[1] - t[0]  # grid spacing in x and t

    # calculate the value of lambda
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number

    # Create the a_cn and b_cn matrices
    a_cn = create_tri_diag_mat(mx - 1, -lmbda / 2, 1 + lmbda, -lmbda / 2)
    b_cn = create_tri_diag_mat(mx - 1, lmbda / 2, 1 - lmbda, lmbda / 2)
    u_j, u_jp1 = np.zeros(x.size), np.zeros(x.size)  # u at current and next time step

    # Set initial condition
    for i in range(0, mx + 1):
        u_j[i] = u_i_func(x[i])

    for j in range(0, mt):
        # Forward Euler time step at inner mesh points
        # PDE discretised at position x[i], time t[j]
        sliced_rhs = b_cn.dot(np.vstack(u_j[1:-1]))

        u_j = spsolve(a_cn, sliced_rhs)
        # Boundary conditions
        u_j = np.append(u_j, bc_L)
        u_j = np.insert(u_j, 0, bc_0, axis=0)

    return x, u_j


def fe_matrix_vector_form(u_i_func, mx, mt, kappa, L, T, bc_0_func, bc_L_func, bc_type):
    """
    A function which performs the forward Euler scheme in matrix/vector form on the heat equation
    :param u_i_func: Function which defines the prescribed initial temperature
    :param mx: Number of grid points in space
    :param mt: Number of grid points in time
    :param kappa: Diffusion constant
    :param L: Length of the spacial domain
    :param T: Total time to solve for
    :param bc_0_func: A function which dictates the boundary condition at x = 0
    :param bc_L_func: A function which dictates the boundary condition at x = L
    :return: Solution of PDE at time T
    """

    # Set up the numerical environment variables
    x, t = np.linspace(0, L, mx + 1), np.linspace(0, T, mt + 1)    # mesh points in space and time
    deltax, deltat = x[1] - x[0], t[1] - t[0]  # grid spacing in x and t

    # calculate the value of lambda, stability requires 0 < lambda < 0.5
    lmbda = kappa * deltat / (deltax ** 2)  # mesh fourier number

    # Check lambda is within the stable range
    if lmbda <= 0 or lmbda >= 0.5:
        raise ValueError(f"lmbda: {lmbda} is not within the range 0 < lmbda < 0.5")

    u_j, u_jp1 = np.zeros(x.size), np.zeros(x.size)  # u at current and next time step

    # Set initial condition
    for i in range(0, mx + 1):
        u_j[i] = u_i_func(x[i])

    if bc_type == 'dirichlet':

        # create the A_FE tridiagonal matrix
        a_fe = create_tri_diag_mat(mx - 1, lmbda, 1 - (2 * lmbda), lmbda)

        for j in range(0, mt):
            # Forward Euler time step at inner mesh points
            # PDE discretised at position x[i], time t[j]
            bc_0 = bc_0_func(t[j])
            bc_L = bc_L_func(t[j])

            zero_vec = np.zeros(mx - 3)
            dir_bc_vec = np.append(zero_vec, bc_L)
            dir_bc_vec = np.insert(dir_bc_vec, 0, bc_0, axis=0)

            u_j = a_fe.dot(np.vstack(u_j[1:-1])) + lmbda * np.vstack(dir_bc_vec)
            # Boundary conditions
            u_j = np.append(u_j, bc_L)
            u_j = np.insert(u_j, 0, bc_0, axis=0)

    elif bc_type == 'neumann':

        # create the A_FE tridiagonal matrix
        a_fe = create_tri_diag_mat(mx + 1, lmbda, 1 - (2 * lmbda), lmbda)

        a_fe[0, 1] = a_fe[0, 1] * 2
        a_fe[-1, -2] = a_fe[-1, -2] * 2

        for j in range(0, mt):
            # Forward Euler time step at inner mesh points
            # PDE discretised at position x[i], time t[j]
            bc_0 = bc_0_func(t[j])
            bc_L = bc_L_func(t[j])

            zero_vec = np.zeros(mx - 1)
            neu_bc_vec = np.append(zero_vec, bc_L)
            neu_bc_vec = np.insert(neu_bc_vec, 0, bc_0, axis=0)

            u_j = a_fe.dot(np.vstack(u_j)) + (2 * lmbda * deltax) * np.vstack(neu_bc_vec)

    return x, u_j


def pde_solver(u_i_func, mx, mt, kappa, L, T, bc_0, bc_L, bc_type, method='fe matrix vector'):
    """
    Function which solves a PDE using the inputted method
    :param u_i_func: Function which defines the prescribed initial temperature
    :param mx: Number of grid points in space
    :param mt: Number of grid points in time
    :param kappa: Diffusion constant
    :param L: Length of the spacial domain
    :param T: Total time to solve for
    :param bc_0: Boundary condition at x = 0
    :param bc_L: Boundary condition at x = L
    :param bc_type: The type of boundary conditions (only the fe matrix vector form method can
                    handle non-zero boundary conditions)
    :param method: Chosen method to use to solve the PDE
    :return: Solution of PDE at time T
    """

    # Check that the boundary conditions given are functions
    if callable(bc_0):
        # Check that the boundary condition functions return a float or integer
        if not isinstance(bc_0(0), (int, np.int_, float, np.float_)):
            raise TypeError(f"bc_0(0): {bc_0(0)} is not an integer or float")

    else:
        raise TypeError(f"bc_0: '{bc_0}' must be a callable function.")

    if callable(bc_L):
        if not isinstance(bc_L(0), (int, np.int_, float, np.float_)):
            raise TypeError(f"bc_L(0): {bc_L(0)} is not an integer or float")

    else:
        raise TypeError(f"bc_L: '{bc_L}' must be a callable function.")

    # Check that L is a float or integer
    if not isinstance(L, (int, np.int_, float, np.float_)):
        raise TypeError(f"L: {L} is not an integer or float")

    # Check that mx and mt are integers
    if not isinstance(mx, (int, np.int_)):
        raise TypeError(f"mx: {mx} is not an integer")
    if not isinstance(mt, (int, np.int_)):
        raise TypeError(f"mt: {mt} is not an integer")

    # Check that kappa is a float or integer
    if not isinstance(kappa, (int, np.int_, float, np.float_)):
        raise TypeError(f"kappa: {kappa} is not an integer or float")

    if callable(u_i_func):
        # Check the output of u_i_func is an integer or float
        u_i_L = u_i_func(L)
        if not isinstance(u_i_L, (int, np.int_, float, np.float_)):
            raise TypeError(f"u_i_L: {u_i_L} must be a float or integer")
        if method == 'forward_euler':
            x, u_j = forward_euler(u_i_func, mx, mt, kappa, L, T, bc_0(0), bc_L(0))
        elif method == 'fe matrix vector':
            x, u_j = fe_matrix_vector_form(u_i_func, mx, mt, kappa, L, T, bc_0, bc_L, bc_type)
        elif method == 'be matrix vector':
            x, u_j = be_matrix_vector_form(u_i_func, mx, mt, kappa, L, T, bc_0(0), bc_L(0))
        elif method == 'crank nicholson':
            x, u_j = c_n(u_i_func, mx, mt, kappa, L, T, bc_0(0), bc_L(0))

        else:
            raise NameError(f"method : {method} isn't present (must select 'forward_euler', "
                            f"'fe matrix vector', 'be matrix vector' or 'crank nicholson')")
    else:
        raise TypeError(f"u_i_func: {u_i_func} is not a callable function")
    return x, u_j


def error_with_time(u_i, u_exact, mt_values):
    """
    Function which measures how the error of the methods changes as mt changes
    :param u_i: Function which defines the prescribed initial temperature
    :param u_exact: exact solution of u
    :param mt_values: Values of mt
    """
    # Set problem parameters/functions
    kappa = 1.0  # diffusion constant - how easily diffusion occurs
    T, mx, L = 0.5, 8, 1.0  # total time to solve for and number of spatial values, length of spatial domain

    # Set up the numerical environment variables
    x = np.linspace(0, L, mx + 1)  # mesh points in space
    mt_count, mt_num = 0, len(mt_values)

    u_j_cn_error, u_j_fe_error, u_j_be_error = [0] * mt_num, [0] * mt_num, [0] * mt_num
    deltat_list = [0] * mt_num

    def bc_0(t):
        return 0

    def bc_L(t):
        return 0

    for mt in mt_values:
        # Ensure mt is an integer
        mt = int(mt)

        # Set up the numerical environment variables
        t = np.linspace(0, T, mt + 1)  # mesh points in space and time
        deltat = t[1] - t[0]  # grid spacing in x and t

        x_fe, u_j_fe = pde_solver(u_i, mx, mt, kappa, L, T, bc_0, bc_L, 'dirichlet')
        x_be, u_j_be = pde_solver(u_i, mx, mt, kappa, L, T, bc_0, bc_L, 'dirichlet', 'be matrix vector')
        x_cn, u_j_cn = pde_solver(u_i, mx, mt, kappa, L, T, bc_0, bc_L, 'dirichlet', 'crank nicholson')
        u_j_exact = u_exact(x, T)

        u_j_fe_error[mt_count] = sum(abs(u_j_fe - u_j_exact)) / len(u_j_exact)
        u_j_be_error[mt_count] = sum(abs(u_j_be - u_j_exact)) / len(u_j_exact)
        u_j_cn_error[mt_count] = sum(abs(u_j_cn - u_j_exact)) / len(u_j_exact)

        deltat_list[mt_count] = deltat
        mt_count += 1

    plt.plot(deltat_list, u_j_cn_error, 'r-', label='Crank Nicholson')
    plt.plot(deltat_list, u_j_fe_error, 'b-', label='Forward Euler')
    plt.plot(deltat_list, u_j_be_error, 'g-', label='Backward Euler')
    plt.legend()
    plt.xlabel(''r'$\Delta t$'), plt.ylabel('Error in u approximation')
    plt.show()


def error_with_x(u_i, u_exact, mx_values):
    """
    Function which measures how the error of the methods changes as mt changes
    :param u_i: Function which defines the prescribed initial temperature
    :param u_exact: exact solution of u
    :param mx_values: Values of mx
    """
    # Set problem parameters/functions
    kappa = 1.0  # diffusion constant - how easily diffusion occurs
    T, mt, L = 0.5, 50000, 1.0  # total time to solve for and number of spatial values, length of spatial domain

    # Set up the numerical environment variables

    mx_count, mx_num = 0, len(mx_values)

    u_j_cn_error, u_j_fe_error, u_j_be_error = [0] * mx_num, [0] * mx_num, [0] * mx_num
    deltax_list = [0] * mx_num

    def bc_0(t):
        return 0

    def bc_L(t):
        return 0

    for mx in mx_values:
        # Ensure mx is an integer
        mx = int(mx)

        # Set up the numerical environment variables
        x = np.linspace(0, T, mx + 1)  # mesh points in space and time
        deltax = x[1] - x[0]  # grid spacing in x and t

        x_fe, u_j_fe = pde_solver(u_i, mx, mt, kappa, L, T, bc_0, bc_L, 'dirichlet', 'fe matrix vector')
        x_be, u_j_be = pde_solver(u_i, mx, mt, kappa, L, T, bc_0, bc_L, 'dirichlet', 'be matrix vector')
        x_cn, u_j_cn = pde_solver(u_i, mx, mt, kappa, L, T, bc_0, bc_L, 'dirichlet', 'crank nicholson')
        u_j_exact = u_exact(x, T)

        u_j_fe_error[mx_count] = sum(abs(u_j_fe - u_j_exact)) / len(u_j_exact)
        u_j_be_error[mx_count] = sum(abs(u_j_be - u_j_exact)) / len(u_j_exact)
        u_j_cn_error[mx_count] = sum(abs(u_j_cn - u_j_exact)) / len(u_j_exact)

        deltax_list[mx_count] = deltax
        mx_count += 1

    plt.plot(deltax_list, u_j_cn_error, 'r-', label='Crank Nicholson')
    plt.plot(deltax_list, u_j_fe_error, 'b-', label='Forward Euler')
    plt.plot(deltax_list, u_j_be_error, 'g-', label='Backward Euler')
    plt.legend()
    plt.xlabel(''r'$\Delta x$'), plt.ylabel('Error in u approximation')
    plt.show()


def main():

    def u_i(x, p=1):
        # initial temperature distribution
        y = (np.sin(pi * x / L)) ** p
        return y

    def u_exact(x, t):
        # the exact solution
        y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
        return y

    def bc_is_0(t):
        return 0

    def bc_is_1(t):
        return 1

    # Set problem parameters/functions
    kappa = 1.0  # diffusion constant - how easily diffusion occurs
    L = 1.0  # length of spatial domain
    T, mt, mx = 0.5, 1000, 10  # total time to solve for and number of time and spatial values

    x, u_j = pde_solver(u_i, mx, mt, 1.0, L, T, bc_is_0, bc_is_0, 'dirichlet', 'forward_euler')

    x_fe, u_j_fe = pde_solver(u_i, mx, mt, 1.0, L, T, bc_is_0, bc_is_0, 'dirichlet', 'fe matrix vector')

    x_be, u_j_be = pde_solver(u_i, mx, mt, 1.0, L, T, bc_is_0, bc_is_0, 'dirichlet', 'be matrix vector')

    x_cn, u_j_cn = pde_solver(u_i, mx, mt, 1.0, L, T, bc_is_0, bc_is_0, 'dirichlet', 'crank nicholson')

    # mt_vals, mx_vals = np.linspace(65, 115, 50), np.linspace(10, 200, 40)
    # error_with_time(u_i, u_exact, mt_vals)
    # error_with_x(u_i, u_exact, mx_vals)

    # Check that forward euler and the matrix vector form return the same answer
    print('Do Forward Euler and matrix vector form return the same u values : ' + str(np.allclose(u_j, u_j_fe)))

    xx = np.linspace(0, L, 250)

    # Plot the final result and exact solution
    # plt.plot(x, u_j_cn, 'bo', label='crank nicholson')
    # plt.plot(x, u_j_be, 'go', label='Backward Euler')
    # plt.plot(x, u_j, 'go', label='FE step wise')
    # plt.plot(x, u_j_fe, 'ro', label='Forward Euler')
    plt.plot(xx, u_exact(xx, T), 'b-', label='exact')
    plt.xlabel('x'), plt.ylabel('u(x,' + str(T) + ')')
    plt.legend(loc='upper right')
    plt.show()

    # x_fe_ne, u_j_fe_ne = pde_solver(u_i, mx, mt, 1.0, L, T, bc_is_1, bc_is_1, 'neumann', 'fe matrix vector')
    # plt.plot(x_fe_ne, u_j_fe_ne, 'bo', label='Forward Euler')
    # plt.xlabel('x'), plt.ylabel('u(x,' + str(T) + ')')
    # plt.legend(loc='upper right')
    # plt.show()


if __name__ == '__main__':
    main()
