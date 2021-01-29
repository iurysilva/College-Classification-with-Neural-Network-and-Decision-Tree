from arquiteturas_rede_neural.arquiteturas import *
from bancos_de_dados.bd_tratado import *

arquitetura = criar_arquitetura_bandeiras()
num_epocas = 500
learning_rate = 0.2
coluna_alvo = 'continente'  # continente, Vinho, vidro, iris

rede = arquitetura
rede.learning_rate = learning_rate
rede.insere_sinapses_e_bias()
banco = rede.banco
base_treino, base_teste, tipos_saidas = tratar_bd(banco, coluna_alvo)
base_treino = base_treino.sample(frac=1).reset_index(drop=True)
rede.aprender(num_epocas, base_treino)
rede.testar(tipos_saidas, base_teste)
