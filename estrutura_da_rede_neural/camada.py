import numpy as np

# Objeto camada, representando todas as possíveis camadas da arquitetura,
# seja ela de entrada, oculta ou final. Possui atributos  para armazenar
# valores importantes como número de neurônios, os valores presentes nos
# neurônios, as sinapses que a conectam com a próxima camada, o bias e o
# erro que são utilizados no Feedfoward e no Backpropagation.
class Camada:
    def __init__(self, numero_neuronios, final=False):
        self.numero_neuronios = numero_neuronios
        self.neuronios = np.zeros((numero_neuronios, 1))
        self.sinapses = None
        self.final = final
        self.bias = np.zeros((numero_neuronios, 1))
        self.erro = None

    # Este método serve para que possamos modificar o número de neurônios da
    # camada, como alguns atributos dependem dessa variável, criamos este
    # método para atualizar tanto o número de neurônios, quanto outros atributos
    # que dependem dele.
    def atualiza_numero_de_neuronios(self, numero_neuronios):
        self.numero_neuronios = numero_neuronios
        self.neuronios = np.zeros((numero_neuronios, 1))
        self.bias = np.zeros((numero_neuronios, 1))
