import pandas as pd

colunas = open('bancos_de_dados/colunas.txt', 'r')
nome_colunas = [coluna[:-1] for coluna in colunas]
colunas.close()
bd_bandeiras = pd.read_csv('bancos_de_dados/banco_bandeiras.csv', names=nome_colunas, header=None)
