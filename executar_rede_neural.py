from estrutura_da_rede_neural import Rede_Neural
from bancos_de_dados import bd_bandeiras
import numpy as np

atributos_entrada = np.array([1, 3])
atributos_saida = np.array([6])
num_de_camadas = 3
num_de_neuronios_por_camada_oculta = 3
linhas_para_aprender = 1


rede = Rede_Neural(atributos_entrada, atributos_saida, num_de_camadas, num_de_neuronios_por_camada_oculta, bd_bandeiras)
rede.insere_sinapses()
rede.aprender(linhas_para_aprender)
