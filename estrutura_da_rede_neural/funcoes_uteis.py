import numpy as np


def sigmoide(x):
    return 1/(1+np.exp(-x))


def derivar_sigmoide(x):
    return x * (1 - x)


def tahn(x):
    return 2 / (1 + np.exp(-2*x)) - 1


def multiplicar_matrizes(a, b):
    return np.matmul(a, b)
