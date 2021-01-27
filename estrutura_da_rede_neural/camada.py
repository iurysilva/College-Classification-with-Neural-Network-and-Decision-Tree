import numpy as np


class Camada:
    def __init__(self, numero_neuronios, final=False):
        self.numero_neuronios = numero_neuronios
        self.neuronios = np.zeros((numero_neuronios, 1))
        self.sinapses = None
        self.final = final
