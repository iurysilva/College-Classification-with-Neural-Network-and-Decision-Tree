import numpy as np


class Camada:
    def __init__(self, numero_neuronios, final=False):
        self.numero_neuronios = numero_neuronios
        self.neuronios = np.zeros((numero_neuronios, 1))
        self.sinapses = None
        self.final = final
        self.bias = np.zeros((numero_neuronios, 1))
        self.erro = None

    def atualiza_numero_de_neuronios(self, numero_neuronios):
        self.numero_neuronios = numero_neuronios
        self.neuronios = np.zeros((numero_neuronios, 1))
        self.bias = np.zeros((numero_neuronios, 1))