import numpy as np
from bancos_de_dados import bd_bandeiras
from estrutura_da_rede_neural import Rede_Neural


def criar_arquitetura_1():
    atributos_entrada = np.array([1, 3])
    atributos_saida = np.array([6])
    num_de_camadas = 3
    num_de_neuronios_por_camada_oculta = 3
    rede = Rede_Neural(atributos_entrada, atributos_saida, num_de_camadas, num_de_neuronios_por_camada_oculta,
                       bd_bandeiras)
    return rede
