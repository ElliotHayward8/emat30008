import matplotlib.pyplot as plt
import numpy as np


def f(x, t):
    return x

def euler_step(f, x, t, h):
    x += (h * f(x, t))
    t += h
    return x, t


def solve_to(x1, t1, x2, t2, deltat_max):