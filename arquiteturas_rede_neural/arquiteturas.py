import numpy as np
from bancos_de_dados import bd_bandeiras
from bancos_de_dados import bd_iris
from bancos_de_dados import bd_vidros
from estrutura_da_rede_neural import Rede_Neural


def criar_arquitetura_1():
    atributos_entrada = np.array([1, 2, 3, 4, 6, 7, 8, 9])
    atributos_saida = np.array([10])
    num_de_camadas = 6
    num_de_neuronios_por_camada_oculta = 10
    rede = Rede_Neural(atributos_entrada, atributos_saida, num_de_camadas, num_de_neuronios_por_camada_oculta,
                       bd_vidros)
    return rede


def criar_arquitetura_2():
    atributos_entrada = np.array([2, 5])
    atributos_saida = np.array([1])
    num_de_camadas = 4
    num_de_neuronios_por_camada_oculta = 4
    rede = Rede_Neural(atributos_entrada, atributos_saida, num_de_camadas, num_de_neuronios_por_camada_oculta,
                       bd_bandeiras)
    return rede


def criar_arquitetura_3():
    atributos_entrada = np.array([0, 1, 2, 3])
    atributos_saida = np.array([4])
    num_de_camadas = 4
    num_de_neuronios_por_camada_oculta = 4
    rede = Rede_Neural(atributos_entrada, atributos_saida, num_de_camadas, num_de_neuronios_por_camada_oculta,
                       bd_iris)
    return rede
