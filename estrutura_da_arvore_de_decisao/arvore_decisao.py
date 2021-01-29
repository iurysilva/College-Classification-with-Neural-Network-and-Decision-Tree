import math
from bancos_de_dados.bd_tratado import tratar_bd
from collections import Counter


class No(object):
    def __init__(self, nome, folha=False):
        self.nome = nome
        self.folha = folha
        self.filhos = []

class Arvore_Decisao(object):
        
    def __init__(self, banco_de_dados, coluna_alvo, amostra=False):
        
        base_treino, base_teste, tipos_saidas = tratar_bd(banco_de_dados,coluna_alvo)
        
        self.numero_amostras = base_treino.shape[0]
        
        if not amostra:
            self.bd = base_treino
            self.alvo = banco_de_dados[coluna_alvo]
        else:
            self.bd = base_treino.sample(self.amostra, random_state=42)
            self.alvo = banco_de_dados[coluna_alvo].sample(self.amostra, random_state=42)
        
        self.entropia_bd = self.calcula_entropia([contagem['freq'] for contagem in self.contagem(self.alvo).values()])
        self.ganhos = {}
        self.razao_ganhos = {}
        
        self.arvore = self.cria_arvore()
        
        for atributo in base_treino.columns[1:]:
            self.calcula_ganho(atributo)
            self.calcula_razao_ganho(atributo)
            
    
    def cria_arvore(self):
        pass
    
    
    def contagem(self, series):
        contagem_dict = {}
        for index, item in enumerate(series):
            if item not in contagem_dict:
                contagem_dict[item] = {'freq': 1, 'indexes': [index]}
            else: 
                contagem_dict[item]['freq'] += 1
                contagem_dict[item]['indexes'].append(index)
        return contagem_dict
    
    
    def contagem_atributo(self, series):
        contagem_dict = {}
        for index, item in enumerate(series):
            if item not in contagem_dict:
                contagem_dict[item] = {'freq': 1, self.alvo[index]: 1}
            elif self.alvo[index] not in contagem_dict[item]:
                contagem_dict[item][self.alvo[index]] = 1
                contagem_dict[item]['freq'] += 1
            else:
                contagem_dict[item][self.alvo[index]] += 1
                contagem_dict[item]['freq'] += 1
        return contagem_dict
            
    
    def calcula_entropia(self, valores):

        entropia = 0
        for contagem in valores:
            cardinalidade = (contagem / self.numero_amostras)
            entropia += - cardinalidade * math.log(cardinalidade, 2)
            #print('-({}/{})log({}/{}) = {}'.format(contagem, self.numero_amostras,contagem, self.numero_amostras,
            #                                       - cardinalidade * math.log(cardinalidade, 2)))
        return entropia
        
    
    def calcula_ganho(self, atributo):
        
        contagem_atributo = self.contagem_atributo(self.bd[f'{atributo}'])
        ganho_atributo = 0
        
        for contagem in contagem_atributo.values():
            cardinalidade = (contagem['freq'] / self.numero_amostras)
            ganho_atributo += cardinalidade * self.calcula_entropia([*contagem.values()][1:])
            
        ganho_atributo = self.entropia_bd - ganho_atributo
        self.ganhos[atributo] = ganho_atributo
    
    
    def calcula_razao_ganho(self, atributo):
        
        contagem_atributo = self.contagem_atributo(self.bd[f'{atributo}'])
        frequencia_atributo = [[*contagem.values()][0] for contagem in contagem_atributo.values()]
        divisao_informacao = self.calcula_entropia(frequencia_atributo)
        
        razao_ganho = self.ganhos[atributo] / divisao_informacao
        self.razao_ganhos[atributo] = razao_ganho


    def melhor_atributo(self,dicionario):
        assert type(dicionario) == dict, "input not dict"

        max_key = max(dicionario, key=lambda k: dicionario[k])
        
        return max_key