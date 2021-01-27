import math
from collections import Counter

class Arvore_Decisao(object):
    
    def __init__(self,banco_de_dados,numero_amostras):
        
        self.numero_amostras = numero_amostras
        self.bd = banco_de_dados.drop(columns=['name','religion']).head(self.numero_amostras)
        self.alvo = banco_de_dados['religion'].head(self.numero_amostras)
        
        self.entropia_bd = 0
        self.calcula_entropia()
        
    
    def calcula_entropia(self):
        contagem_religioes = self.contagem(self.alvo)
        self.entropia_bd = 0
        for contagem in contagem_religioes.values():
            cardionalidade = (contagem / self.numero_de_amostras)
            self.entropia += -cardionalidade*math.log(cardionalidade,2)
            print('-({}/{})log({}/{}) = {}'.format(contagem,self.numero_amostras,contagem,self.numero_amostras,
                                                -cardionalidade*math.log(cardionalidade,2)))

    def contagem(self, series):
        contagem_dict = {}
        for index, item in enumerate(series):
            if item not in contagem_dict:
                contagem_dict[item] = {freq: 1, indexes: [index]}
            else: 
                contagem_dict[item][freq] += 1
                contagem_dict[item][indexes].append(index)
        return contagem_dict