import numpy as np


def sigmoide(x):
    return 1/(1+np.exp(-x))
