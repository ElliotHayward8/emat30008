import matplotlib.pyplot as plt
import warnings
import numpy as np
from numerical_shooting import shooting
from scipy.optimize import fsolve
from scipy.linalg import norm
from value_checks import array_int_or_float


def cubic(x, pars):
    """
    This function defines a cubic equation
    :param x: Value of x
    :param pars: Defines the additional parameter c
    :return: returns the value of the cubic equation at x
    """
    c = pars[0]
    return x ** 3 - x + c


def normal_hopf(u0, t, pars):
    """
    Function which defines the Hopf bifurcation normal form system of ODEs
    :param u0: Parameter values (u1, u2)
    :param t: Time value
    :param pars: Additional parameters which are required to define the system of ODEs (beta, sigma)
    :return: returns an array of du1/dt and du2/dt at (X, t) as a numpy array
    """
    beta, sigma = pars[0], pars[1]
    u1, u2 = u0[0], u0[1]

    du1dt = beta * u1 - u2 + (sigma * u1) * (u1 ** 2 + u2 ** 2)
    du2dt = u1 + beta * u2 + (sigma * u2) * (u1 ** 2 + u2 ** 2)
    return np.array([du1dt, du2dt])


# phase condition for the normal hopf bifurcation (u1 gradient = 0)
def pc_normal_hopf(u0, t, pars):
    return normal_hopf(u0, pars)[0]


def modified_hopf(u0, t, pars):
    """
    Function which defines the Hopf bifurcation normal form system of ODEs
    :param u0: Parameter values (u1, u2)
    :param t: Time value
    :param pars: Additional variables which are required to define the system of ODEs
    :return: returns an array of du1/dt and du2/dt at (X, t) as a numpy array
    """
    beta = pars[0]
    u1, u2 = u0[0], u0[1]

    du1dt = (beta * u1) - u2 + u1 * (u1 ** 2 + u2 ** 2) - u1 * ((u1 ** 2 + u2 ** 2) ** 2)
    du2dt = u1 + (beta * u2) + u2 * (u1 ** 2 + u2 ** 2) - u2 * ((u1 ** 2 + u2 ** 2) ** 2)
    return np.array([du1dt, du2dt])


def nat_par_continuation(f, u0_guess, pars0, max_par, vary_par, max_steps=100, discretisation=shooting,
                         solver=fsolve, phase_cond='none'):
    """
    Function which performs natural parameter continuation on an inputted function/ODE (f)
    :param f: An ODE/function to perform natural parameter continuation on
    :param u0_guess: Estimated value of the solution at the initial parameter values
    :param pars0: The initial variables
    :param max_par: Maximum value of the varying parameter
    :param vary_par: The index position of the parameter which is varying
    :param max_steps: Maximum number of steps to take
    :param discretisation: The type of discretisation to use
    :param solver: The solver to use (ensure that your desired solver is imported as the name inputted here)
    :param phase_cond: Phase condition for the inputted ODE/system of ODEs
    :return: A list of values of the varied parameter and a list of solution values
    """

    # cancel any RuntimeWarnings as they just inform that the iteration isn't making good progress
    warnings.simplefilter("ignore", category=RuntimeWarning)

    # define the minimum value of the parameter, and create a list of values of the varying parameter
    min_par = pars0[vary_par]

    par_list = np.linspace(min_par, max_par, max_steps)

    u0 = u0_guess
    # For every parameter value find the solution
    sol_list = []
    for par in par_list:
        pars0[vary_par] = par

        # if a phase condition is required, pass it into the pars so that it can be passed into the solver
        if phase_cond != 'none':
            initial_pars0 = (phase_cond, pars0)
        else:
            initial_pars0 = pars0

        sol = np.array(solver(discretisation(f), u0, args=initial_pars0))

        sol_list.append(sol)

        # Set the previous solution to be the guess for the next parameter value
        u0 = sol

    return par_list, np.array(sol_list)


def pseudo_arclength(f, u0_guess, pars0, max_par, vary_par, max_steps=100, discretisation=shooting,
                     solver=fsolve, phase_cond='none'):
    """
    Function which performs natural parameter continuation on an inputted function/ODE (f)
    :param f: An ODE/function to perform natural parameter continuation on
    :param u0_guess: Estimated value of the solution at the initial parameter values
    :param pars0: The initial values of the variables
    :param max_par: Maximum value of the parameter which is being varied
    :param vary_par: The index position of the parameter which is varying within pars0
    :param max_steps: Maximum number of steps to take
    :param discretisation: The type of discretisation to use
    :param solver: The solver to use (ensure that your desired solver is imported as the name inputted here)
    :param phase_cond: Phase condition for the inputted ODE/system of ODEs
    :return: A list of values of the varied parameter and a list of solution values
    """

    def pseudo_eq(u2_guess, u2, secant):
        """
        Function which defines the Pseudo-Arclength equation
        :param u2_guess: Predicted solution value using the secant
        :param u2: Solution value
        :param secant: Calculated secant ((x1, a1) - (x0, a0))
        :return: Returns the pseudo arclength equation
        """
        return np.dot(u2 - u2_guess, secant)

    def new_F(u0a, f,  discretisation, secant, vary_par, pars, phase_cond):
        """
        Redefine the root finding problem so that it also includes the pseudo-arclength equation
        :param u2_guess: Predicted solution value using the secant
        :param f: An ODE/function to perform natural parameter continuation on
        :param u0a: Combination of the x value and alpha value
        :param discretisation: The type of discretisation to use
        :param secant: Calculated secant ((x1, a1) - (x0, a0))
        :param vary_par: The index position of the parameter which is varying within pars
        :param pars: The values of any additional variables
        :param phase_cond: Phase condition for the problem (is 'none' if no phase condition is required)
        :return: Root finding problem to be inputted into the solver
        """
        disc_f = discretisation(f)

        if phase_cond == 'none':
            original = disc_f(u0a[:-1], pars)
        else:
            original = disc_f(u0a[:-1], phase_cond, pars)

        pseudo_arc_eq = pseudo_eq(u2_guess, u0a[:-1], secant)

        return np.append(original, pseudo_arc_eq)

    # Cancel any RuntimeWarnings as they just inform that the iteration isn't making good progress
    warnings.simplefilter("ignore", category=RuntimeWarning)

    # Define the minimum value of the parameter, and create a list of values of the varying parameter
    min_par = pars0[vary_par]
    par_list = np.linspace(min_par, max_par, max_steps)

    # Define a function which is True or False depending on whether the final alpha value has been reached, this
    # function is different depending on whether max_par is greater than or less than min_par
    if min_par < max_par:
        def final_alpha(alpha, max_par):
            if alpha >= max_par:
                return False
            else:
                return True
    else:
        def final_alpha(alpha, max_par):
            if alpha <= max_par:
                return False
            else:
                return True

    u0 = u0_guess

    # If a phase condition is required, pass it into the pars so that it can be passed into the solver
    if phase_cond != 'none':
        initial_pars0 = (phase_cond, pars0)
    else:
        initial_pars0 = pars0

    # Obtain the first solution using u0
    u1 = np.array(solver(discretisation(f), u0, args=initial_pars0))

    pars0[vary_par] = par_list[1]

    # If a phase condition is required, pass it into the pars so that it can be passed into the solver
    if phase_cond != 'none':
        initial_pars0 = (phase_cond, pars0)
    else:
        initial_pars0 = pars0

    # Obtain the second solution using u1
    u2 = np.array(solver(discretisation(f), u1, args=initial_pars0))

    # Define the list of solutions and alpha values
    sol_list = [u1[0], u2[0]]
    alpha_list = [par_list[0], par_list[1]]

    i = 0
    run = True

    while run:

        pars = pars0

        # Obtain the previous two alpha and x values to calculate the secant
        a_0, a_1 = alpha_list[-2], alpha_list[-1]
        x_0, x_1 = sol_list[-2], sol_list[-1]

        # Calculate the change in x and alpha (a)
        delta_x = x_1 - x_0
        delta_a = a_1 - a_0
        secant = np.append(delta_x, delta_a)

        # Calculate the predicted values of x and alpha (a)
        pred_x = x_1 + delta_x
        pred_a = a_1 + delta_a
        pred_val = np.append(pred_x, pred_a)

        pars[vary_par] = pred_a

        full_sol = np.array(solver(new_F, pred_val, args=(f, discretisation, secant, pars, phase_cond)))

        # Split the full_sol into the alpha and U values
        print(str(full_sol) + ' is the full solution')
        sol = full_sol[:-1]
        alpha = full_sol[-1]

        # Append the solution and alpha values to a list that is eventually returned by the function
        sol_list.append(sol)
        alpha_list.append(alpha)

        i += 1
        run = final_alpha(alpha, max_par)

    return alpha_list, sol_list


def num_continuation(f, method, u0_guess, pars0, max_par, vary_par, max_steps=100, discretisation=shooting,
                     solver=fsolve, phase_cond='none'):
    """
    Function which performs numerical continuation on a function, f, over an inputted range of alpha values
    :param f: An ODE/function to perform natural parameter continuation on
    :param method: The chosen numerical continuation method ('natural' or 'pseudo')
    :param u0_guess: Estimated value of the solution at the initial parameter values
    :param pars0: The initial values of the variables
    :param max_par: Maximum value of the parameter which is being varied
    :param vary_par: The index position of the parameter which is varying within pars0
    :param max_steps: Maximum number of steps to take
    :param discretisation: The type of discretisation to use
    :param solver: The solver to use (ensure that your desired solver is imported as the name inputted here)
    :param phase_cond: Phase condition for the inputted ODE/system of ODEs
    :return: A list of values of the varied parameter and a list of solution values
    """

    # Check u0_guess, pars0 only contain integers or floats
    array_int_or_float(u0_guess, 'u0_guess')
    array_int_or_float(pars0, 'pars0')

    # Check that f is a callable function
    if not callable(f):
        raise TypeError('The f function must be a callable function')
    # Check that the output of the discretisation of f is a function
    if not callable(discretisation(f)):
        raise TypeError('The discretisation of the f function must be callable')
    else:
        func_d = discretisation(f)

        # Check that the output is an array of integers/floats
        if phase_cond == 'none':
            trial_func_val = func_d(u0_guess, pars0)
        else:
            trial_func_val = func_d(u0_guess, phase_cond, pars0)

        array_int_or_float(trial_func_val, 'trial_func_val')

    # Check that max_par is an integer or float
    if not isinstance(max_par, (int, float, np.float_, np.int_)):
        raise TypeError(f'max_par: {max_par} must be an integer or a float')

    # Check that max_steps is a positive integer
    if not isinstance(max_steps, (int, np.int_)):
        raise TypeError(f'max_steps: {max_steps} must be an integer')
    else:
        if max_steps <= 0:
            raise ValueError(f'max_steps: {max_steps} must be a positive integer')

    # Check that vary_par is 0 or a positive integer
    if not isinstance(vary_par, (int, np.int_)):
        raise TypeError(f'vary_par: {vary_par} must be an integer')
    elif vary_par >= len(pars0):
        raise ValueError(f'vary_par ({vary_par}) is out of range of pars0, value should be between 0 and '
                             f'{len(pars0) - 1}')
    elif vary_par < 0:
        raise ValueError(f'vary_par: {vary_par} < 0, but it must be > 0 ')

    # Check that the method is a string
    if not isinstance(method, str):
        raise TypeError(f'method ({method}) must be a string')

    if method == 'natural':
        par_list, sol_list = nat_par_continuation(f, u0_guess, pars0, max_par, vary_par, max_steps, discretisation,
                                                  solver, phase_cond)
    elif method == 'pseudo':
        par_list, sol_list = pseudo_arclength(f, u0_guess, pars0, max_par, vary_par, max_steps, discretisation,
                                              solver, phase_cond)
    else:
        raise NameError(f"method : {method} isn't present (must select 'natural' or 'pseudo')")

    return par_list, sol_list


def main():
    """
    First example on an algebraic cubic equation (not an ODE, no phase condition required)

    Try both continuation methods on the cubic function when varying the c parameter between 2 and -2
    """

    u0_guess_cubic = np.array([1])

    np_par_list, np_sol_list = num_continuation(cubic, 'natural', u0_guess_cubic, [-2], 2, 0, 200, lambda x: x, fsolve)

    pa_par_list, pa_sol_list = pseudo_arclength(cubic, u0_guess_cubic, [-2], 2, 0, 200, lambda x: x, fsolve)

    # Plot a graph of c against the norm of x (only one value in x so it is already the norm)
    plt.plot(np_par_list, np_sol_list, 'b-', label='Natural parameter')
    # plt.plot(pa_par_list, pa_sol_list, 'r-', label='Pseudo-arclength')
    plt.xlabel('c'), plt.ylabel('||x||'), plt.legend()
    plt.show()

    """
    Second example on the hopf bifurcation normal form
    
    Try both continuation methods on the hopf bifurcation system of ODEs, varying the Beta parameter between 2 and 0
    """

    u0_guess_hopfnormal = np.array([1.4, 0, 6.3])

    # nat_par_continuation(normal_hopf, u0_guess_hopfnormal, [2, -1], 2, 0, 100, shooting, fsolve, pc_normal_hopf)

    """
    Third example on the modified Hopf bifurcation normal form 
    
    Try both continuation methods on the modified Hopf bifurcation system of ODEs, varying the Beta parameter between 2
    and -1
    """


if __name__ == '__main__':
    main()
