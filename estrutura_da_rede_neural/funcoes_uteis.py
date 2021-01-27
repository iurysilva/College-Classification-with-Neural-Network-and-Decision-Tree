import numpy as np


def sigmoide(x):
    return 1/(1+np.exp(-x))


def multiplicar_matrizes(a, b):
    return np.matmul(a, b)
