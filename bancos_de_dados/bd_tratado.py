from bancos_de_dados.ferramentas import *
import numpy as np
import pandas as pd


def tratar_bd(banco, coluna):

    tipos_saidas = banco[coluna].unique()
    base_treino = np.array([])
    base_teste = np.array([])
    tamanho_treino = 0
    tamanho_teste = 0
    
    for classe in tipos_saidas:
        banco_auxiliar = banco.query('%s==%d' %(coluna, classe))
        treino_auxiliar = retorna_treino(banco_auxiliar)
        for linha in (treino_auxiliar.values):
            base_treino = np.concatenate((base_treino, linha), axis=0)
        teste_auxiliar = retorna_teste(banco_auxiliar, treino_auxiliar)
        for linha in (teste_auxiliar.values):
            base_teste = np.concatenate((base_teste, linha), axis=0)
        tamanho_treino += len(treino_auxiliar)
        tamanho_teste += len(teste_auxiliar)
        
    num_colunas = len(banco.columns)
    base_treino = np.reshape(base_treino, (tamanho_treino, num_colunas))
    base_teste = np.reshape(base_teste, (tamanho_teste, num_colunas))
    base_treino = pd.DataFrame(base_treino, columns=banco.columns)
    base_teste = pd.DataFrame(base_teste, columns=banco.columns)

    return base_treino, base_teste, tipos_saidas
