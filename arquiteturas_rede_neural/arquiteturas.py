import numpy as np
from bancos_de_dados import bd_bandeiras
from bancos_de_dados import bd_iris
from estrutura_da_rede_neural import Rede_Neural


def criar_arquitetura_1():
    atributos_entrada = np.array([1, 2])
    atributos_saida = np.array([6])
    num_de_camadas = 3
    num_de_neuronios_por_camada_oculta = 3
    rede = Rede_Neural(atributos_entrada, atributos_saida, num_de_camadas, num_de_neuronios_por_camada_oculta,
                       bd_bandeiras)
    return rede


def criar_arquitetura_2():
    atributos_entrada = np.array([1, 2, 6])
    atributos_saida = np.array([5])
    num_de_camadas = 6
    num_de_neuronios_por_camada_oculta = 5
    rede = Rede_Neural(atributos_entrada, atributos_saida, num_de_camadas, num_de_neuronios_por_camada_oculta,
                       bd_bandeiras)
    return rede


def criar_arquitetura_3():
    atributos_entrada = np.array([0, 1, 2, 3])
    atributos_saida = np.array([4])
    num_de_camadas = 4
    num_de_neuronios_por_camada_oculta = 3
    rede = Rede_Neural(atributos_entrada, atributos_saida, num_de_camadas, num_de_neuronios_por_camada_oculta,
                       bd_iris)
    return rede
