from estrutura_da_rede_neural import Rede_Neural
from bancos_de_dados import bd_bandeiras

num_de_entradas = 2
num_de_saidas = 1
num_de_camadas = 3
num_de_neuronios_por_camada_oculta = 3


rede = Rede_Neural(2, 1, 3, 3, bd_bandeiras)
rede.insere_sinapses()
rede.mostra_informacoes_das_camadas()
rede.aprender(136)
