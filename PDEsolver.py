# simple forward Euler solver for the 1D heat equation
#   u_t = kappa u_xx  0<x<L, 0<t<T
# with zero-temperature boundary conditions
#   u=0 at x=0,L, t>0
# and prescribed initial temperature
#   u=u_I(x) 0<=x<=L,t=0
import warnings

import numpy as np
import matplotlib.pyplot as plt
from math import pi
import scipy
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


def fe_matrix_vector_form(u_i_func, mx, mt, kappa, L, T, bc_0_func, bc_L_func, bc_type, source):
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
    :param bc_type: The type of boundary conditions
    :param source: Defines any source term within the PDE, if there is no source term it is None
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
            bc_0 = bc_0_func(0, t[j])
            bc_L = bc_L_func(L, t[j])

            # Define the RHS function
            if source is not None:
                F_x = np.vstack(source(x[1:-1], t[j]))
            else:
                F_x = np.vstack(0 * x[1:-1])

            zero_vec = np.zeros(mx - 3)
            dir_bc_vec = np.append(zero_vec, bc_L)
            dir_bc_vec = np.insert(dir_bc_vec, 0, bc_0, axis=0)

            u_j = a_fe.dot(np.vstack(u_j[1:-1])) + (lmbda * np.vstack(dir_bc_vec)) + (deltat * F_x)
            # Boundary conditions
            u_j = np.append(u_j, bc_L)
            u_j = np.insert(u_j, 0, bc_0, axis=0)

            # Define the RHS function
            if source is not None:
                F_x = np.vstack(source(x[1:-1], t[j]))
            else:
                F_x = np.vstack(0 * x[1:-1])

    elif bc_type == 'neumann':

        # create the A_FE tridiagonal matrix
        a_fe = create_tri_diag_mat(mx + 1, lmbda, 1 - (2 * lmbda), lmbda)

        a_fe[0, 1] = a_fe[0, 1] * 2
        a_fe[mx, mx - 1] = a_fe[mx, mx - 1] * 2
        for j in range(0, mt):
            # Forward Euler time step at inner mesh points
            # PDE discretised at position x[i], time t[j]
            bc_0 = -bc_0_func(0, t[j])
            bc_L = bc_L_func(L, t[j])
            # Define the RHS function
            if source is not None:
                F_x = np.vstack(source(x, t[j]))
            else:
                F_x = np.vstack(0 * x)

            zero_vec = np.zeros(mx - 1)
            neu_bc_vec = np.append(zero_vec, bc_L)
            neu_bc_vec = np.insert(neu_bc_vec, 0, -bc_0, axis=0)
            u_j = a_fe.dot(np.vstack(u_j)) + ((2 * lmbda * deltat) * np.vstack(neu_bc_vec)) + (deltat * F_x)

    elif bc_type == 'periodic':

        # create the A_FE tridiagonal matrix
        a_fe = create_tri_diag_mat(mx, lmbda, 1 - (2 * lmbda), lmbda)

        a_fe[0, -1] = lmbda
        a_fe[-1, 0] = lmbda

        for j in range(0, mt):
            # Forward Euler time step at inner mesh points
            # PDE discretised at position x[i], time t[j]
            bc_0 = bc_0_func(0, t[j])
            bc_L = bc_L_func(L, t[j])

            if source is not None:
                F_x = np.vstack(source(x[:-1], t[j]))
            else:
                F_x = np.vstack(0 * x[:-1])

            u_j = a_fe.dot(np.vstack(u_j[:-1])) + (deltat * F_x)

            # Set u_j_L = u_j_0
            u_j = np.append(u_j, u_j[0])

    return x, u_j


def pde_solver(u_i_func, mx, mt, kappa, L, T, bc_0, bc_L, bc_type='dirichlet', method='fe matrix vector', source=None):
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
    :param source: Defines any source term within the PDE, if there is no source term it is None
    :return: Solution of PDE at time T
    """

    # cancel SciPy warnings about changing sparse matrices
    warnings.simplefilter("ignore", category=scipy.sparse.SparseEfficiencyWarning)

    # Check that the boundary conditions given are functions
    if callable(bc_0):
        # Check that the boundary condition functions return a float or integer
        if not isinstance(bc_0(0, 0), (int, np.int_, float, np.float_)):
            raise TypeError(f"bc_0(0): {bc_0(0, 0)} is not an integer or float")

    else:
        raise TypeError(f"bc_0: '{bc_0}' must be a callable function.")

    if callable(bc_L):
        if not isinstance(bc_L(L, 0), (int, np.int_, float, np.float_)):
            raise TypeError(f"bc_L(0): {bc_L(L, 0)} is not an integer or float")

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

    # Check that source is a callable function that is a an integer or float when called with an x and t value
    if source is not None:
        if callable(source):
            if not isinstance(source(0, 0), (int, np.int_, float, np.float_)):
                raise TypeError(f"source: {source(0, 0)} must be a float or intger")
        else:
            raise TypeError(f"source: {source} is not a callable function")

    if callable(u_i_func):
        # Check the output of u_i_func is an integer or float
        u_i_L = u_i_func(L)
        if not isinstance(u_i_L, (int, np.int_, float, np.float_)):
            raise TypeError(f"u_i_L: {u_i_L} must be a float or integer")
        if method == 'forward_euler':
            if bc_type != 'dirichlet':
                raise TypeError('forward_euler method only works for dirichlet boundary conditions')
            x, u_j = forward_euler(u_i_func, mx, mt, kappa, L, T, bc_0(0, 0), bc_L(L, 0))
        elif method == 'fe matrix vector':
            x, u_j = fe_matrix_vector_form(u_i_func, mx, mt, kappa, L, T, bc_0, bc_L, bc_type, source)
        elif method == 'be matrix vector':
            x, u_j = be_matrix_vector_form(u_i_func, mx, mt, kappa, L, T, bc_0(0, 0), bc_L(L, 0))
        elif method == 'crank nicholson':
            x, u_j = c_n(u_i_func, mx, mt, kappa, L, T, bc_0(0, 0), bc_L(L, 0))

        else:
            raise NameError(f"method : {method} isn't present (must select 'forward_euler', "
                            f"'fe matrix vector', 'be matrix vector' or 'crank nicholson')")
    else:
        raise TypeError(f"u_i_func: {u_i_func} is not a callable function")
    return x, u_j


def main():

    """
    First example: Solve a simple PDE using all 3 methods
    """

    def u_i(x, p=1):
        # initial temperature distribution to use for all 3 methods
        y = (np.sin(pi * x / L)) ** p
        return y

    def u_exact(x, t):
        # the exact solution of the first example
        y = np.exp(-kappa * (pi ** 2 / L ** 2) * t) * np.sin(pi * x / L)
        return y

    """
    Define some boundary conditions, here we will use u(0, t) = u(L, t) = 0 for the simple example
    """

    def bc_is_0(x, t):
        return 0

    def bc_is_1(x, t):
        return 1

    """
    Define the problem parameters for the simple PDE used in example 1
    """

    kappa = 1.0
    L = 1.0
    T, mt, mx = 0.5, 1000, 10

    x_fe, u_j_fe = pde_solver(u_i, mx, mt, 1.0, L, T, bc_is_0, bc_is_0, 'dirichlet', 'fe matrix vector')

    x_be, u_j_be = pde_solver(u_i, mx, mt, 1.0, L, T, bc_is_0, bc_is_0, 'dirichlet', 'be matrix vector')

    x_cn, u_j_cn = pde_solver(u_i, mx, mt, 1.0, L, T, bc_is_0, bc_is_0, 'dirichlet', 'crank nicholson')

    # Check that forward euler and the matrix vector form return the same answer
    x, u_j = pde_solver(u_i, mx, mt, 1.0, L, T, bc_is_0, bc_is_0, 'dirichlet', 'forward_euler')

    print('Do Forward Euler and matrix vector form return the same u values : ' + str(np.allclose(u_j, u_j_fe)))

    xx = np.linspace(0, L, 250)

    # Plot the final result and exact solution
    plt.plot(x, u_j_cn, 'g-', label='crank nicholson')
    plt.plot(x, u_j_be, 'm-', label='Backward Euler')
    plt.plot(x, u_j_fe, 'r-', label='Forward Euler')
    plt.plot(xx, u_exact(xx, T), 'b-', label='exact')
    plt.xlabel('x'), plt.ylabel('u(x,' + str(T) + ')')
    plt.legend(loc='upper right')
    plt.show()

    L = 2
    T = 0.5
    mx = 100
    mt = 10000

    def source_term(x, t):
        return x + t

    # Plot Neumann boundary conditions
    x_fe_ne, u_j_fe_ne = pde_solver(u_i, mx, mt, 1.0, L, T, bc_is_0, bc_is_1, 'neumann', 'fe matrix vector', source_term)
    plt.plot(x_fe_ne, u_j_fe_ne, 'b-', label='Forward Euler')
    plt.xlabel('x'), plt.ylabel('u(x,' + str(T) + ')')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    main()
