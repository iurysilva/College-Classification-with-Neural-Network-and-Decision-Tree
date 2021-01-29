from arquiteturas_rede_neural.arquiteturas import *
from bancos_de_dados.bd_tratado import *

arquitetura = criar_arquitetura_vinho()
num_epocas = 500
learning_rate = 0.5

rede = arquitetura
rede.learning_rate = learning_rate
rede.insere_sinapses_e_bias()
coluna_alvo = rede.banco.columns[rede.atributos_de_saida[0]]  # continente, Vinho, vidro, iris
banco = rede.banco
base_treino, base_teste, tipos_saidas = tratar_bd(banco, coluna_alvo)
base_treino = base_treino.sample(frac=1).reset_index(drop=True)
rede.aprender(num_epocas, base_treino)
matriz = rede.testar(tipos_saidas, base_teste)
calcula_resultados(matriz)
