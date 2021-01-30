from estrutura_da_arvore_de_decisao import Arvore_Decisao, Pergunta
from bancos_de_dados import bd_vinho
from bancos_de_dados.bd_tratado import tratar_bd

banco = bd_vinho
coluna_alvo = 'Vinho'

base_treino, base_teste, tipos_saidas = tratar_bd(banco,coluna_alvo)

ad = Arvore_Decisao(base_treino,coluna_alvo)
print('{}\n{}/{}'.format(ad.raiz,ad.raiz.filho_esq,ad.raiz.filho_dir))