import math
import pandas as pd
import numpy as np


class NoDecisao(object):
    """Registra a partição da árvore em dois novos ramos,
    tendo a referência da pergunta e dos dois nós filhos.    
    """
    def __init__(self, pergunta, ramo_true, ramo_false):
        self.pergunta = pergunta
        self.ramo_true = ramo_true
        self.ramo_false = ramo_false


class Arvore_Decisao(object):
        
    def __init__(self, banco_de_dados, coluna_alvo):
        
        self.bd = banco_de_dados
        self.alvo = banco_de_dados[coluna_alvo]
        self.numero_amostras = banco_de_dados.shape[0]
        self.numero_atributos = banco_de_dados.shape[1]
        
        self.entropia_bd = self.calcula_entropia([contagem['freq'] for contagem in self.contagem(self.alvo).values()])
        
        self.ganhos = {}
        self.razao_ganhos = {}

        self.cria_arvore()
    
    
    def cria_arvore(self):
        
        for coluna in self.bd.columns[1:]:
            self.calcula_ganho(coluna)
        
        melhor_atributo = self.melhor_atributo(self.ganhos)

        self.cria_arvore_recursiva(self.bd)
    
    
    def cria_arvore_recursiva(self, atributo):
        
        melhor_ganho = 0 
        melhor_pergunta = None
        
        incerteza_atual = self.calcula_entropia([contagem['freq'] for contagem in self.contagem(atributo).values()])
        n_atributos = self.numero_atributos

        for coluna in range(n_atributos):
            
            valores = np.unique(atributo.values)

            for valor in valores:  # for each value

                pergunta = Pergunta(coluna, valor)

                linha_true, linha_false = self.particao(atributo, pergunta)

                if len(linha_true) == 0 or len(linha_false) == 0:
                    continue

                ganho = self.ganho_info(linha_true, linha_false, atributo)

                '''
                if ganho >= melhor_ganho:
                    melhor_ganho, melhor_pergunta = ganho, pergunta

        return melhor_ganho, melhor_pergunta'''
    
    
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


    def ganho_info(self, esquerda, direita, atributo):
        contagem_esquerda = self.contagem_atributo(esquerda).values()
        contagem_direita = self.contagem_atributo(direita).values()
        contagem_atributo = self.contagem_atributo(atributo).values()
        print(contagem_esquerda)
        p = float(esquerda.shape[0] / self.numero_amostras)
        #esquerda = [*(self.contagem_atributo(esquerda).values())][1:]
        #direita = [*(self.contagem_atributo(direita).values()).values()][1:]]
        #atributo = [*(self.contagem_atributo(atributo).values()).values()][1:]
        #return self.calcula_entropia(atributo) - p * self.calcula_entropia(esquerda) - (1 - p) * self.calcula_entropia(direita)


    def melhor_atributo(self,dicionario):

        max_key = max(dicionario, key=lambda k: dicionario[k])
        
        return max_key

        
    def particao(self, dataset, pergunta):
        """Particiona um dataset.
        Para cada linha do dataset, faz a checagem em relação à pergunta.
        Se der verdadeiro, a linha é adicionada ao 'linha_true', senão,
        é adicionada ao 'linha_false'.
        """
        linha_true, linha_false = [], []
        for _, row in dataset.iterrows():
            if pergunta.match(row):
                linha_true.append(row)
            else:
                linha_false.append(row)
        return pd.DataFrame(linha_true), pd.DataFrame(linha_false)


    def acha_melhor_particao(self, dataset):
        """Find the best pergunta to ask by iterating over every feature / value
        and calculating the information gain."""
        melhor_ganho = 0  # keep track of the best information gain
        melhor_pergunta = None  # keep train of the feature / value that produced it
        incerteza_atual = self.calcula_entropia(dataset)
        n_atributos = len(dataset[0]) - 1  # number of columns

        for col in range(n_atributos):  # for each feature

            values = set([row[col] for row in dataset])  # unique values in the column

            for val in values:  # for each value

                pergunta = Pergunta(col, val)

                # particiona o dataset
                linha_true, linha_false = self.particao(dataset, pergunta)

                # Skip this split if it doesn't divide the
                # dataset.
                if len(linha_true) == 0 or len(linha_false) == 0:
                    continue

                # Calculate the information gain from this split
                ganho = self.ganho_info(linha_true, linha_false, col)

                
                if ganho >= melhor_ganho:
                    melhor_ganho, melhor_pergunta = ganho, pergunta

        return melhor_ganho, melhor_pergunta


class Pergunta(object):
    """A pergunta é usada para particionar o dataset.

    This class just records a 'column number' (e.g., 0 for Color) and a
    'column value' (e.g., Green). The 'match' method is used to compare
    the feature value in an example to the feature value stored in the
    pergunta. See the demo below.
    """

    def __init__(self, coluna, valor):
        self.coluna = coluna
        self.valor = valor

    def match(self, linha):
        # Compare the feature valor in an linha to the
        # feature valor in this pergunta.
        val = linha[self.coluna]
        return val >= self.valor
        
    '''
    def __repr__(self):
        # This is just a helper method to print
        # the pergunta in a readable format.
        condicao = ">="
        
        return "Is %s %s %s?" % (
            self.bd[self.coluna], condicao, str(self.valor))'''