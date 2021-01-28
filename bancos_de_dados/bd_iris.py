import pandas as pd

colunas = open('bancos_de_dados/colunas_iris.txt', 'r')
nome_colunas = [coluna[:-1] for coluna in colunas]
colunas.close()
bd_iris = pd.read_csv('bancos_de_dados/banco_iris.csv', names=nome_colunas, header=None)
