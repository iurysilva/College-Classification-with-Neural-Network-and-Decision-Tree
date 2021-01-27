import numpy as np


def sigmoide(x):
    return 1/(1+np.exp(-x))

def derivar_sigmoide(x):
    return x * (1 - x)

def multiplicar_matrizes(a, b):
    return np.matmul(a, b)
