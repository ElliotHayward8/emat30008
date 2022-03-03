import matplotlib.pyplot as plt
import numpy as np


def array_int_or_float(var, var_name):
    """
    A function which checks if an array contains only integer or float values (for example for t_eval)
    :param var: Variable to check the values of
    :param var_name: Name of the variable
    """

    is_not_int = np.array(var).dtype != np.int_
    is_not_float = np.array(var).dtype != np.float_
    if is_not_int and is_not_float:
        raise TypeError(f"'{var_name}' : '{var}' contains invalid types. "
                        f"{var_name} should contain only integers and/or floats.")


def ode_checker(f, x0, t_eval, *vars):
    """
    A function which checks that the ode specified is a callable function and that it has outputs which match the
    size/shape of the initial conditions. If these conditions aren't met an error message which corresponds to the
    correct error will be raised
    :param f: ODE to be checked
    :param x0: Initial conditions
    :param t_eval: Specified list of time values
    :param vars: Array of any additional variables
    """
    if callable(f):

        # Test the output of the ODE (f) is the same as x0
        test_t = t_eval[0]
        test_x1 = f(x0, test_t, *vars)

        if isinstance(test_x1, (int, float, np.int_, np.float_, list, np.ndarray)):
            if not np.array(test_x1).shape == np.array(x0).shape:
                raise ValueError("Incorrect shape of ODE or x0 (shape of x0 and output of f aren't the same)")
        else:
            raise TypeError(f"Output of f is of the type {type(test_x1)}. It should be an int, float, list or array")

    else:
        raise TypeError(f"f: '{f}' must be a callable function.")

