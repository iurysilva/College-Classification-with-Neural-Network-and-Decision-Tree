import pandas as pd
import numpy as np

colunas = open('bancos_de_dados/colunas_vinho.txt', 'r')
nome_colunas = [coluna[:-1] for coluna in colunas]
colunas.close()
bd_vinho = pd.read_csv('bancos_de_dados/banco_vinho.csv',names=nome_colunas, header=None)
print(bd_vinho)