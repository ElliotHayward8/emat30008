import matplotlib.pyplot as plt
import numpy as np

def int_or_float(par, par_name):
    """
    A function which determines if a parameter is an integer/float
    :param par: Parameter to check
    :param par_name: Name of the parameter to check
    """

    if not isinstance(par, (int, float, np.float_, np.int_)):
        raise TypeError(f'{par_name}: {par} must be an integer or float')

def pos_int(par, par_name):
    """
    A function which determines if a parameter is a positive integer
    :param par: Parameter to check
    :param par_name: Name of the parameter to check
    """

    if not isinstance(par, (int, np.int_)):
        raise TypeError(f'{par_name}: {par} must be an integer')

    if par <= 0:
        raise ValueError(f'{par_name}: {par} must be a positive integer')

def pos_int_or_float(par, par_name):
    """
    A function which determines if a parameter is a positive integer/float
    :param par: Parameter to check
    :param par_name: Name of the parameter to check
    """

    if not isinstance(par, (int, float, np.float_, np.int_)):
        raise TypeError(f'{par_name}: {par} must be an integer or float')

    if par <= 0:
        raise ValueError(f'{par_name}: {par} must be a positive integer or float')


def array_int_or_float(par, par_name):
    """
    A function which checks if a parameter contains only integer or float values (for example for t_eval)
    :param par: Parameter to check the values of
    :param par_name: Name of the parameter
    """

    is_not_int = np.array(par).dtype != np.int_
    is_not_float = np.array(par).dtype != np.float_
    if is_not_int and is_not_float:
        raise TypeError(f"'{par_name}' : '{par}' contains invalid types. "
                        f"{par_name} should contain only integers and floats.")


def ode_checker(f, x0, t_eval, *pars):
    """
    A function which checks that the ode specified is a callable function and that it has outputs which match the
    size/shape of the initial conditions. If these conditions aren't met an error message which corresponds to the
    correct error is raised
    :param f: ODE to be checked
    :param x0: Initial conditions
    :param t_eval: Specified list of time values
    :param pars: Array of any additional parameters
    """
    if callable(f):

        if pars != ():
            array_int_or_float(pars, 'pars')

        # Test the output of the ODE (f) is the same as x0
        test_t = t_eval[0]
        test_x1 = f(x0, test_t, *pars)

        if isinstance(test_x1, (int, float, np.int_, np.float_, list, np.ndarray)):
            if not np.array(test_x1).shape == np.array(x0).shape:
                raise ValueError("Incorrect shape of ODE or x0 (shape of x0 and output of f aren't the same)")
        else:
            raise TypeError(f"Output of f is of the type {type(test_x1)}. It should be an int, float, list or array")

    else:
        raise TypeError(f"f: '{f}' must be a callable function.")
