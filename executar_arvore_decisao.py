from estrutura_da_arvore_de_decisao import ArvoreDecisao
from bancos_de_dados import bd_vinho
from bancos_de_dados.bd_tratado import tratar_bd
from bancos_de_dados.ferramentas import calcula_resultados
import numpy as np

banco = bd_vinho
coluna_alvo = 'Vinho'
n_execucoes = 10

acuracias = {'Base Teste': [], 'Base Treino': [], 'Base Total': []}

for execucao in range(n_execucoes):

    base_treino, base_teste, tipos_saidas = tratar_bd(banco, coluna_alvo)
    dict_bancos = {'Base Teste': base_teste, 'Base Treino': base_treino, 'Base Total': banco}

    ad = ArvoreDecisao(base_treino, coluna_alvo)

    for nome_banco in list(dict_bancos.keys()):

        predicao = ad.classifica(dict_bancos[nome_banco])
        matriz_confusao = np.zeros((3, 3))

        soma = 0

        for _, linha in predicao.iterrows():
            valor_predicao = int(linha[1]) - 1
            valor_real = int(linha[0]) - 1

            matriz_confusao[valor_predicao][valor_real] += 1

        acuracias[nome_banco].append(calcula_resultados(matriz_confusao,True))

acuracia_bdteste = f"Desvio Padrão das Acurácias da base de teste: {np.std(acuracias['Base Teste']):.2f} // Média das Acurácias da base de teste: {np.mean(acuracias['Base Teste']):.2f}%"
acuracia_bdtreino = f"Desvio Padrão das Acurácias da base de treino: {np.std(acuracias['Base Treino']):.2f} // Média das Acurácias da base de treino: {np.mean(acuracias['Base Treino']):.2f}%"
acuracia_bdtotal = f"Desvio Padrão das Acurácias da base total: {np.std(acuracias['Base Total']):.2f} // Média das Acurácias da base total: {np.mean(acuracias['Base Total']):.2f}%"

print(acuracia_bdteste)
print(acuracia_bdtreino)
print(acuracia_bdtotal)
