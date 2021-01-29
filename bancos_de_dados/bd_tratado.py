from bancos_de_dados.bd_vinho import bd_vinho
from bancos_de_dados.ferramentas import *
import numpy as np
import pandas as pd

def tratar_bd(banco):
    tipos_saidas = bd_vinho['Wine'].unique()
    base_treino = np.array([])
    base_teste = np.array([])
    for classe in (tipos_saidas):
        banco_auxiliar = banco.query('Wine==%d' %classe)
        treino_auxiliar = retorna_treino(banco_auxiliar)
        #print(len(treino_auxiliar))
        for linha in (treino_auxiliar.values):
            base_treino = np.concatenate((base_treino, linha), axis=0)
        teste_auxiliar = retorna_teste(banco_auxiliar, treino_auxiliar)
        for linha in (teste_auxiliar.values):
            base_teste = np.concatenate((base_teste, linha), axis=0)

    base_treino = np.reshape(base_treino, (125, 14))
    base_teste = np.reshape(base_teste, (53, 14))
    base_treino = pd.DataFrame(base_treino)
    base_teste = pd.DataFrame(base_teste)

    return base_treino, base_teste
