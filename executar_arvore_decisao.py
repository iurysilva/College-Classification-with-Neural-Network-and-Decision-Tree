from estrutura_da_arvore_de_decisao import ArvoreDecisao
from bancos_de_dados import bd_vinho
from bancos_de_dados.bd_tratado import tratar_bd
from bancos_de_dados.ferramentas import calcula_resultados
import time
import numpy as np

banco = bd_vinho
coluna_alvo = 'Vinho'
n_execucoes = 10

# Calculamos os tempos de execução do algoritmo.
tempos = np.array([])

# Criamos um dicionário a fim de armazenar as acurácias de cada
# execução, e cada banco testado.
acuracias = {'Base Teste': [], 'Base Treino': [], 'Base Total': []}

for execucao in range(n_execucoes):

    tempo_inicial = time.perf_counter()

    # A base de dados é tratada.
    base_treino, base_teste, tipos_saidas = tratar_bd(banco, coluna_alvo)
    dict_bancos = {'Base Teste': base_teste, 'Base Treino': base_treino, 'Base Total': banco}

    # Instanciamos a árvore de decisão, informando o alvo.
    ad = ArvoreDecisao(base_treino, coluna_alvo)

    # ad.imprime(ad.raiz, '  ')
    # print('Altura: ', ad.altura())

    for nome_banco in list(dict_bancos.keys()):

        # Aqui é feita a classificação da base fornecida.
        predicao = ad.classifica(dict_bancos[nome_banco])

        # Uma matriz vazia é gerada, a ser preenchida em breve.
        matriz_confusao = np.zeros((3, 3))

        soma = 0

        # O cálculo da matriz de confusão é feito.
        for _, linha in predicao.iterrows():
            valor_predicao = int(linha[1]) - 1
            valor_real = int(linha[0]) - 1

            matriz_confusao[valor_predicao][valor_real] += 1

        # Os resultados obtidos são calculados.
        acuracias[nome_banco].append(calcula_resultados(matriz_confusao, True))

    # Por fim, o tempo de execução.
    tempos = np.append(tempos, time.perf_counter() - tempo_inicial)    

acuracia_bdteste = f"Desvio Padrão das Acurácias da base de teste: {np.std(acuracias['Base Teste']):.3f} // Média das Acurácias da base de teste: {np.mean(acuracias['Base Teste']):.3f}%"
acuracia_bdtreino = f"Desvio Padrão das Acurácias da base de treino: {np.std(acuracias['Base Treino']):.3f} // Média das Acurácias da base de treino: {np.mean(acuracias['Base Treino']):.3f}%"
acuracia_bdtotal = f"Desvio Padrão das Acurácias da base total: {np.std(acuracias['Base Total']):.3f} // Média das Acurácias da base total: {np.mean(acuracias['Base Total']):.3f}%"

# Finalmente, imprimimos a média das acurácias de cada banco testado, assim como
# a média dos devios padrões e também a média do tempo de todas as execuções.
print('{:.3f}'.format(np.mean(tempos)))
print(acuracia_bdteste)
print(acuracia_bdtreino)
print(acuracia_bdtotal)
