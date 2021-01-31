from estrutura_da_arvore_de_decisao import Arvore_Decisao, Pergunta, Folha
from bancos_de_dados import bd_vinho
from bancos_de_dados.bd_tratado import tratar_bd
from bancos_de_dados.ferramentas import calcula_resultados
import numpy as np

banco = bd_vinho
coluna_alvo = 'Vinho'

acuracias = {'Base Teste':[],'Base Treino':[], 'Base Total':[]}

for i in range(10):
    
    base_treino, base_teste, tipos_saidas = tratar_bd(banco, coluna_alvo)
    dic_bancos = {'Base Teste':base_teste,'Base Treino':base_treino, 'Base Total':banco}
    
    ad = Arvore_Decisao(base_treino, coluna_alvo)
    
    for nome_banco in list(dic_bancos.keys()):

        predicao = ad.classifica(dic_bancos[nome_banco])
        matriz_confusao = np.zeros((3,3))

        soma = 0

        for _, linha in predicao.iterrows():
            valor_predicao = int(linha[1])-1
            valor_real = int(linha[0])-1

            matriz_confusao[valor_predicao][valor_real] += 1

        acuracias[nome_banco].append(calcula_resultados(matriz_confusao))

acuracia_bdteste = [np.std(acuracias['Base Teste']),np.mean(acuracias['Base Teste'])]
acuracia_bdtreino = [np.std(acuracias['Base Treino']),np.mean(acuracias['Base Treino'])]
acuracia_bdtotal = [np.std(acuracias['Base Total']),np.mean(acuracias['Base Total'])]
print(acuracia_bdteste)
print(acuracia_bdtreino)
print(acuracia_bdtotal)