from arquiteturas_rede_neural import criar_arquitetura_4
from bancos_de_dados.bd_tratado import *

arquitetura = criar_arquitetura_4()
num_epocas = 500

rede = arquitetura
rede.learning_rate = 0.5
rede.insere_sinapses_e_bias()
base_treino, base_teste, tipos_saidas = tratar_bd(rede.banco)
rede.aprender(num_epocas, base_treino)
