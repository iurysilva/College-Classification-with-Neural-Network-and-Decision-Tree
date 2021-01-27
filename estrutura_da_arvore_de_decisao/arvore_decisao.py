import math
from collections import Counter


class Arvore_Decisao(object):
        
    def __init__(self, banco_de_dados, numero_amostras):
        
        self.numero_amostras = numero_amostras
        self.bd = banco_de_dados.drop(columns=['name','religion']).head(self.numero_amostras)
        self.alvo = banco_de_dados['religion'].head(self.numero_amostras)
        
        self.entropia_bd = self.calcula_entropia([contagem['freq'] for contagem in self.contagem(self.alvo).values()])
        self.ganhos = {}
        self.razao_ganhos = {}
        
        for atributo in ['stripes','bars','circles','crosses']:
            self.calcula_ganho(atributo)
            self.calcula_razao_ganho(atributo)
    
    
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