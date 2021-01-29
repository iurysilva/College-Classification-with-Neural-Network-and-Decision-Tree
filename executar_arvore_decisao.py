from estrutura_da_arvore_de_decisao import Arvore_Decisao
from bancos_de_dados import bd_bandeiras
from bancos_de_dados import bd_vinho

ad = Arvore_Decisao(bd_vinho,'Vinho')
print('Entropia:{}'.format(ad.entropia_bd))
#print('Atributos:{}'.format(ad.ganhos))
print('Razao Ganho:{}'.format(ad.razao_ganhos))
