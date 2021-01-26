import pandas as pd

colunas = open('bancos_de_dados/colunas.txt', 'r')
lista = [coluna[:-1] for coluna in colunas]
colunas.close()
bd_bandeiras = pd.read_csv('bancos_de_dados/banco1.csv', names=lista, header=None)
