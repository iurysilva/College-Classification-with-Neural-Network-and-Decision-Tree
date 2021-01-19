import pandas as pd

colunas = open('csv/columns.txt','r')
lista = [coluna[:-1] for coluna in colunas]
colunas.close()

df = pd.read_csv('csv/flag.data.csv',names=lista, header=None)

print(df.head(10))