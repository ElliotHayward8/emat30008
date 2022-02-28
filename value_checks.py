import matplotlib.pyplot as plt
import numpy as np


def array_int_or_float(var, var_name):
    """
    A function which checks if an array contains only integer or float values (for example for t_eval)
    :param var: Variable to check the values of
    :param var_name: Name of the variable
    :return:
    """

    is_not_int = np.array(var).dtype != np.int_
    is_not_float = np.array(var).dtype != np.float_
    if is_not_int and is_not_float:
        raise TypeError(f"'{var_name}' : '{var}' contains invalid types. "
                        f"{var_name} should contain only integers and/or floats.")


