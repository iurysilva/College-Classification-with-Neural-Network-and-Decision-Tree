from arquiteturas_rede_neural.arquiteturas import *
from bancos_de_dados.bd_tratado import *
import time
import numpy as np

arquitetura = criar_arquitetura_vinho()
num_epocas = 500
learning_rate = 1
num_execucoes = 10

rede = arquitetura
rede.learning_rate = learning_rate
rede.insere_sinapses_e_bias()
coluna_alvo = rede.banco.columns[rede.atributos_de_saida[0]]  # continente, Vinho, vidro, iris
banco = rede.banco
tempos = np.array([])

acuracia_treino = np.zeros(num_execucoes)
acuracia_teste = np.zeros(num_execucoes)
acuracia_total = np.zeros(num_execucoes)
for execucao in range(num_execucoes):
    print('\nexecução : ', execucao)
    tempo_inicial = time.perf_counter()
    rede = arquitetura
    rede.learning_rate = learning_rate
    rede.insere_sinapses_e_bias()

    base_treino, base_teste, tipos_saidas = tratar_bd(banco, coluna_alvo)
    base_treino = base_treino.sample(frac=1).reset_index(drop=True)
    rede.aprender(num_epocas, base_treino)
    matriz_treino = rede.testar(tipos_saidas, base_treino)
    matriz_teste = rede.testar(tipos_saidas, base_teste)
    matriz_total = rede.testar(tipos_saidas, banco)

    tempos = np.append(tempos, time.perf_counter() - tempo_inicial)

    print('base de treino: ')
    acuracia_treino[execucao] = calcula_resultados(matriz_treino, True)
    print('base de teste: ')
    acuracia_teste[execucao] = calcula_resultados(matriz_teste, True)
    print('base inteira: ')
    acuracia_total[execucao] = calcula_resultados(matriz_total, True)
print('')
print('media de treino: ', np.mean(acuracia_treino))
print('media de teste: ', np.mean(acuracia_teste))
print('media de total: ', np.mean(acuracia_total))
print('')
print('desvio padrão de treino: ', np.std(acuracia_treino))
print('desvio padrão de teste: ', np.std(acuracia_teste))
print('desvio padrão de total: ', np.std(acuracia_total))
print()
print('Tempo médio de execução: ', np.mean(tempos))

