import matplotlib.pyplot as plt
import warnings
import numpy as np
from numerical_shooting import shooting
from scipy.optimize import fsolve, root
import scipy
from value_checks import array_int_or_float, int_or_float, pos_int_or_float


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


# phase condition for the normal Hopf bifurcation (u1 gradient = 0)
def pc_normal_hopf(u0, pars):
    return normal_hopf(u0, 0, pars)[0]


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

# Phase condition for the modified Hopf bifurcation (u1 gradient = 0)
def pc_mod_hopf(u0, pars):
    return modified_hopf(u0, 0, pars)[0]

def return_new_pars(pars, vary_par, new_alpha):
    """
    A function which takes in a new alpha value and replaces the old alpha value within the pars list
    :param pars: List of additional parameters
    :param vary_par: The index position of the parameter which is varying
    :param new_alpha: New alpha value to be subbed into the pars list
    :return: pars with the new alpha value substituted in
    """

    pars[vary_par] = new_alpha
    return pars


def nat_par_continuation(f, u0_guess, pars, max_par, vary_par, max_steps=100, discretisation=shooting,
                         solver=fsolve, phase_cond='none'):
    """
    Function which performs natural parameter continuation on an inputted function/ODE (f)
    :param f: An ODE/function to perform natural parameter continuation on
    :param u0_guess: Estimated value of the solution at the initial parameter values
    :param pars: The initial variables
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
    min_par = pars[vary_par]
    par_list = np.linspace(min_par, max_par, max_steps)

    u0 = u0_guess
    # For every parameter value find the solution
    sol_list = []
    for par in par_list:
        pars[vary_par] = par

        # If a phase condition is required, pass it into the pars so that it can also be passed into the solver
        if phase_cond != 'none':
            initial_pars0 = (phase_cond, pars)
        else:
            initial_pars0 = pars

        sol = np.array(solver(discretisation(f), u0, args=initial_pars0))

        sol_list.append(sol)

        # Set the previous solution to be the guess for the next parameter value
        u0 = sol

    return par_list, np.array(sol_list)


def pseudo_arclength_no_pc(f, u0_guess, pars, max_par, vary_par, max_steps=100, discretisation=shooting, solver=fsolve):
    """
    Function which performs natural parameter continuation on an inputted function/ODE (f)
    :param f: An ODE/function to perform natural parameter continuation on
    :param u0_guess: Estimated value of the solution at the initial parameter values
    :param pars: The initial values of the variables
    :param max_par: Maximum value of the parameter which is being varied
    :param vary_par: The index position of the parameter which is varying within pars0
    :param max_steps: Maximum number of steps to take
    :param discretisation: The type of discretisation to use
    :param solver: The solver to use (ensure that your desired solver is imported as the name inputted here)
    :return: A list of values of the varied parameter and a list of solution values
    """

    # Cancel any RuntimeWarnings as they just inform that the iteration isn't making good progress
    warnings.simplefilter("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    # Define the minimum value of the parameter, and create a list of values of the varying parameter
    min_par = pars[vary_par]
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

    # Obtain the first solution using u0
    u1 = np.array(solver(discretisation(f), u0, args=pars))

    pars[vary_par] = par_list[1]

    # Obtain the second solution using u1
    u2 = np.array(solver(discretisation(f), u1, args=pars))

    # Define the list of solutions and alpha values
    sol_list = [u1, u2]
    alpha_list = [par_list[0], par_list[1]]

    i = 0
    run = True

    # Iterates until latest alpha value surpasses max_par
    while run:

        # Obtain the previous two alpha and x values to calculate the secant
        a_0, a_1 = alpha_list[-2], alpha_list[-1]
        x_0, x_1 = sol_list[-2], sol_list[-1]

        # Calculate the change in x and alpha (a)
        delta_x = x_1 - x_0
        delta_a = a_1 - a_0

        # Calculate the predicted values of x and alpha (a)
        pred_x = x_1 + delta_x
        pred_a = a_1 + delta_a
        pred_val = np.append(pred_x, pred_a)

        secant = np.append(delta_x, delta_a)

        # Define the varying variable as the predicted alpha value
        pars[vary_par] = pred_a

        full_sol = np.array(solver(lambda nui: np.append(discretisation(f)(nui[:-1], pars=return_new_pars(pars, vary_par
                                                                                                          , nui[-1])),
                                                         np.dot(nui - pred_val, secant)), pred_val))

        # Split the full_sol into the alpha and U values
        sol = full_sol[:-1]
        alpha = full_sol[-1]
        # Append the solution and alpha values to a list that is eventually returned by the function
        sol_list.append(sol)
        alpha_list.append(alpha)

        i += 1

        run = final_alpha(alpha, max_par)

        # For graphing of results purposes, end the simulation if the value goes below 0
        if len(sol) == 1:
            if sol[0] <= 0:
                print('Single valued solution went below 0 so the simulation was ended')
                run = False

    return alpha_list, sol_list


def pseudo_arclength_pc(f, u0_guess, pars, max_par, vary_par, phase_cond, max_steps=100, discretisation=shooting,
                        solver=fsolve):
    """
    Function which performs natural parameter continuation on an inputted function/ODE (f)
    :param f: An ODE/function to perform natural parameter continuation on
    :param u0_guess: Estimated value of the solution at the initial parameter values
    :param pars: The initial values of the variables
    :param max_par: Maximum value of the parameter which is being varied
    :param vary_par: The index position of the parameter which is varying within pars0
    :param phase_cond: The phase condition required for shooting
    :param max_steps: Maximum number of steps to take
    :param discretisation: The type of discretisation to use
    :param solver: The solver to use (ensure that your desired solver is imported as the name inputted here)
    :return: A list of values of the varied parameter and a list of solution values
    """

    # Cancel any RuntimeWarnings as they just inform that the iteration isn't making good progress
    warnings.simplefilter("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    # Define the minimum value of the parameter, and create a list of values of the varying parameter
    min_par = pars[vary_par]
    par_list = np.linspace(min_par, max_par, max_steps)

    # Define a function which is True or False depending on whether the final alpha value has been reached, this
    # function is different depending on whether max_par is greater than or less than min_par
    if min_par < max_par:
        def final_alpha(alpha, max_par):
            if alpha >= max_par or alpha < min_par:
                return False
            else:
                return True
    else:
        def final_alpha(alpha, max_par):
            if alpha <= max_par or alpha > min_par:
                return False
            else:
                return True

    u0 = u0_guess

    pars_phase_cond = (phase_cond, pars)

    # Obtain the first solution using u0
    u1 = np.array(solver(discretisation(f), u0, args=pars_phase_cond))

    # Redefine the pars variable for the second parameter value
    pars[vary_par] = par_list[1]
    pars_phase_cond = (phase_cond, pars)

    # Obtain the second solution using u1
    u2 = np.array(solver(discretisation(f), u1, args=pars_phase_cond))

    # Define the list of solutions and alpha values
    sol_list = [u1, u2]
    alpha_list = [par_list[0], par_list[1]]

    i = 0
    run = True

    # Iterates until latest alpha value surpasses max_par
    while run:

        # Obtain the previous two alpha and x values to calculate the secant
        a_0, a_1 = alpha_list[-2], alpha_list[-1]
        x_0, x_1 = sol_list[-2], sol_list[-1]

        # Calculate the change in x and alpha (a)
        delta_x = x_1 - x_0
        delta_a = a_1 - a_0

        # Calculate the predicted values of x and alpha (a)
        pred_x = x_1 + delta_x
        pred_a = a_1 + delta_a
        pred_val = np.append(pred_x, pred_a)

        secant = np.append(delta_x, delta_a)

        # Define the varying variable as the predicted alpha value
        pars[vary_par] = pred_a
        full_sol = np.array(solver(lambda nui: np.append(discretisation(f)(nui[:-1], phase_cond, return_new_pars(
            pars, vary_par, nui[-1])),
                                                         np.dot(nui - pred_val, secant)), pred_val))

        # Split the full_sol into the alpha and U values
        sol = full_sol[:-1]
        alpha = full_sol[-1]
        # Append the solution and alpha values to a list that is eventually returned by the function
        sol_list.append(sol)
        alpha_list.append(alpha)

        i += 1

        run = final_alpha(alpha, max_par)

    return alpha_list, sol_list

def num_continuation(f, method, u0_guess, pars, max_par, vary_par, max_steps=100, discretisation=shooting,
                     solver=fsolve, phase_cond='none'):
    """
    Function which performs numerical continuation on a function, f, over an inputted range of alpha values
    :param f: An ODE/function to perform natural parameter continuation on
    :param method: The chosen numerical continuation method ('natural' or 'pseudo')
    :param u0_guess: Estimated value of the solution at the initial parameter values
    :param pars: The initial values of the variables
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
    array_int_or_float(pars, 'pars')

    # Check that f is a callable function
    if not callable(f):
        raise TypeError('The f function must be a callable function')
    # Check that the output of the discretisation of f is a function
    if not callable(discretisation(f)):
        raise TypeError('The discretisation of the f function must be callable')
    else:
        # Check that the output is an array of integers/floats
        if phase_cond == 'none':
            trial_func_val = discretisation(f)(u0_guess, pars)
        else:
            trial_func_val = discretisation(f)(u0_guess, phase_cond, pars)

        array_int_or_float(trial_func_val, 'trial_func_val')

    # Check that max_par is an integer or float
    int_or_float(max_par, 'max_par')

    # Check that max_steps is a positive integer
    if not isinstance(max_steps, (int, np.int_)):
        raise TypeError(f'max_steps: {max_steps} must be an integer')
    else:
        if max_steps <= 0:
            raise ValueError(f'max_steps: {max_steps} must be a positive integer')

    # Check that vary_par is 0 or a positive integer
    if not isinstance(vary_par, (int, np.int_)):
        raise TypeError(f'vary_par: {vary_par} must be an integer')
    elif vary_par >= len(pars):
        raise ValueError(f'vary_par ({vary_par}) is out of range of pars0, value should be between 0 and '
                         f'{len(pars) - 1}')
    elif vary_par < 0:
        raise ValueError(f'vary_par: {vary_par} < 0, but it must be > 0 ')

    # Check that the method is a string
    if not isinstance(method, str):
        raise TypeError(f'method ({method}) must be a string')

    if method == 'natural':
        par_list, sol_list = nat_par_continuation(f, u0_guess, pars, max_par, vary_par, max_steps, discretisation,
                                                  solver, phase_cond)
    elif method == 'pseudo':
        if phase_cond == 'none':
            par_list, sol_list = pseudo_arclength_no_pc(f, u0_guess, pars, max_par, vary_par, max_steps, discretisation,
                                                        solver)
        elif callable(phase_cond):
            par_list, sol_list = pseudo_arclength_pc(f, u0_guess, pars, max_par, vary_par, phase_cond, max_steps,
                                                     discretisation, solver)
        else:
            raise TypeError('If a phase condition is required it must be inputted as a callable function')
    else:
        raise NameError(f'method : {method} isn\'t present (must select \'natural\' or \'pseudo\')')

    return par_list, sol_list


def main():
    """
    First example on an algebraic cubic equation (not an ODE, no phase condition required)

    Try both continuation methods on the cubic function when varying the c parameter between 2 and -2
    """

    u0_guess_cubic = np.array([1])

    # Perform natural parameter continuation with c varying from -2 to 2
    np_par_list, np_sol_list = num_continuation(cubic, 'natural', u0_guess_cubic, [-2], 2, 0, 200, lambda x: x, fsolve)

    # Perform pseudo-arclength continuation with c between -2 and 2 (pseudo-arclength stops if the value of x is < 0)
    pa_par_list, pa_sol_list = num_continuation(cubic, 'pseudo', u0_guess_cubic, [-2], 2, 0, 200, lambda x: x, fsolve)

    # As the solution is a single value the norm is simply the absolute value
    norm_pa_sol_list = [abs(number) for number in pa_sol_list]

    # Plot a graph of c against the norm of x (only one value in x so it is already the norm)
    plt.plot(np_par_list, np_sol_list, 'b-', label='Natural parameter')
    plt.plot(pa_par_list, norm_pa_sol_list, 'r-', label='Pseudo-arclength')
    plt.title('Cubic equation continuation with c varying between -2 and 2')
    plt.xlabel('c'), plt.ylabel('||x||'), plt.legend()
    plt.show()

    """
    Second example on the hopf bifurcation normal form
    
    Try both continuation methods on the hopf bifurcation system of ODEs, varying the Beta parameter between 2 and 0
    """

    u0_guess_hopfnormal = np.array([1.41, 0, 6.28])

    np_par_list, np_sol_list = num_continuation(normal_hopf, 'natural', u0_guess_hopfnormal, [2, -1], -0.4, 0, 100,
                                                shooting, fsolve, pc_normal_hopf)

    pa_par_list, pa_sol_list = num_continuation(normal_hopf, 'pseudo', u0_guess_hopfnormal, [2, -1], -0.4, 0, 100,
                                                shooting, fsolve, pc_normal_hopf)

    # As the solutions from both methods have multiple values the norm must be calculated (excluding the T value)

    norm_np_sol_list = scipy.linalg.norm(np_sol_list[:, :-1], axis=1, keepdims=True)

    count = 0
    while count < len(pa_sol_list):
        pa_sol_list[count] = np.array(scipy.linalg.norm(pa_sol_list[count][:-1], axis=0, keepdims=True))
        count += 1

    # Plot a graph of Beta against the norm of the solution
    plt.plot(np_par_list, norm_np_sol_list, 'b-', label='Natural parameter')
    plt.plot(pa_par_list, pa_sol_list, 'r-', label='Pseudo-arclength')
    plt.xlabel('Beta'), plt.ylabel('||x||'), plt.legend()
    plt.title('Hopf bifurcation normal form continuation varying Beta between 2 and -0.4')
    plt.show()

    # I had to go from 2 to just past 0 (rather than 0 to 2) because the bifurcation occurs at Beta = 0, therefore, the
    # continuation doesn't work when starting at Beta = 0. I then extended the length of the continuation slightly to
    # put further emphasis on the location of the bifurcation

    """
    Third example on the modified Hopf bifurcation normal form 
    
    Try both continuation methods on the modified Hopf bifurcation system of ODEs, varying the Beta parameter between 2
    and -1
    """

    u0_guess_modhopf = [1.41, 0, 6.28]

    np_par_list, np_sol_list = num_continuation(modified_hopf, 'natural', u0_guess_modhopf, [2, -1], -1, 0, 100,
                                                shooting, fsolve, pc_mod_hopf)

    pa_par_list, pa_sol_list = num_continuation(modified_hopf, 'pseudo', u0_guess_modhopf, [2, -1], -1, 0, 100,
                                                shooting, fsolve, pc_mod_hopf)

    # As the solutions from both methods have multiple values the norm must be calculated (excluding the T value)
    norm_np_sol_list = scipy.linalg.norm(np_sol_list[:, :-1], axis=1, keepdims=True)

    count = 0
    while count < len(pa_sol_list):
        pa_sol_list[count] = np.array(scipy.linalg.norm(pa_sol_list[count][:-1], axis=0, keepdims=True))
        count += 1

    # Plot a graph of Beta against the norm of the solution
    plt.plot(np_par_list, norm_np_sol_list, 'b-', label='Natural parameter')
    plt.plot(pa_par_list, pa_sol_list, 'r-', label='Pseudo-arclength')
    plt.title('Modified Hopf bifurcation normal form continuation varying Beta between 2 and -1')
    plt.xlabel('Beta'), plt.ylabel('||x||'), plt.legend()
    plt.show()

    # Observations of the graph show that there is a bifurcation at approximately Beta = -0.2


if __name__ == '__main__':
    main()
