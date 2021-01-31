import math
import pandas as pd
import numpy as np
from collections import Counter
import statistics as st


class ArvoreDecisao(object):
        
    def __init__(self, banco_de_dados, coluna_alvo):
        
        self.banco_de_dados = banco_de_dados

        self.coluna_alvo = coluna_alvo
        self.dados_alvo = self.banco_de_dados[coluna_alvo]

        self.n_linhas = banco_de_dados.shape[0]
        self.n_colunas = banco_de_dados.shape[1]
        self.colunas = banco_de_dados.columns

        self.raiz = None
        
        self.cria_arvore()


    def __repr__(self):
        return 'Linhas:{}\nColunas:{}'.format(self.n_linhas, self.n_colunas)
    
    
    def altura(self):
        return self.altura_recursiva(self.raiz, 0)


    def altura_recursiva(self, no_atual, altura_atual):
        if not no_atual:
            return altura_atual
        altura_esq = self.altura_recursiva(no_atual.filho_esq, altura_atual + 1)
        altura_dir = self.altura_recursiva(no_atual.filho_dir, altura_atual + 1)
        return max(altura_esq, altura_dir)


    def cria_arvore(self):
        self.raiz = self.cria_arvore_recursiva(self.banco_de_dados)


    def cria_arvore_recursiva(self, banco):
        
        no = self.verifica_melhor_corte(banco)

        numero_classes_esq = len(np.unique(no.filho_esq[self.coluna_alvo]))
        numero_classes_dir = len(np.unique(no.filho_dir[self.coluna_alvo]))
        
        if numero_classes_esq != 1:
            no.filho_esq = self.cria_arvore_recursiva(no.filho_esq)
        else:
            no.filho_esq = Folha(no.filho_esq, self.coluna_alvo)

        if numero_classes_dir != 1:
            no.filho_dir = self.cria_arvore_recursiva(no.filho_dir)
        else:
            no.filho_dir = Folha(no.filho_dir, self.coluna_alvo)
            
        return no

    def verifica_melhor_corte(self, banco):
        
        maior_ganho = {'Atributo': '', 'Ganho de Informação': 0, 'Pergunta': None}
        filho_esquerdo = None
        filho_direito = None

        entropia_pai = self.calcula_entropia(banco)

        for coluna in self.colunas[1:]:
            banco_coluna = self.arredonda_float(banco[coluna])

            for linha in banco_coluna[:-1]:
                pergunta = Pergunta(coluna, linha)
                banco_esquerdo, banco_direito = self.corta_banco(banco, pergunta)
                ganho_info = self.calcula_ganho_informacao(banco_esquerdo, banco_direito, entropia_pai)
                if ganho_info > maior_ganho['Ganho de Informação']:
                    maior_ganho = {'Atributo': coluna, 'Ganho de Informação': ganho_info, 'Pergunta': pergunta}
                    filho_esquerdo, filho_direito = banco_esquerdo, banco_direito

        return No(atributo=maior_ganho['Atributo'],
                  entropia=entropia_pai,
                  pergunta=maior_ganho['Pergunta'],
                  filho_esquerdo=filho_esquerdo,
                  filho_direito=filho_direito)


    def calcula_entropia(self, banco):
        
        entropia = 0
        
        numero_ocorrencias_classe = list(Counter(banco[self.coluna_alvo]).values())

        n_linhas = banco.shape[0]
        
        for quantidade in numero_ocorrencias_classe:
            peso = (quantidade / n_linhas)
            entropia += - peso * math.log(peso, 2)
            
        return entropia

    def calcula_ganho_informacao(self, banco_esquerdo, banco_direito, entropia_pai):
        
        ganho_informacao = 0
        
        n_amostras = banco_esquerdo.shape[0] + banco_direito.shape[0]
        peso_esquerdo = banco_esquerdo.shape[0] / n_amostras
        peso_direito = banco_direito.shape[0] / n_amostras
        
        if not banco_esquerdo.empty:
            calculo_esquerdo = peso_esquerdo * self.calcula_entropia(banco_esquerdo)
        else:
            calculo_esquerdo = 0
            
        calculo_direito = peso_direito * self.calcula_entropia(banco_direito)
        
        ganho_informacao = calculo_esquerdo + calculo_direito
        
        return entropia_pai - ganho_informacao

    def corta_banco(self, banco, pergunta):
        
        linha_true, linha_false = [], []
        
        for _, row in banco.iterrows():

            if pergunta.verifica(row):
                linha_true.append(row)

            else:
                linha_false.append(row)
        
        banco_esquerdo = pd.DataFrame(linha_false)
        banco_direito = pd.DataFrame(linha_true)
        
        return banco_esquerdo, banco_direito

    def arredonda_float(self, banco):
        return list(map(int, Counter(['%d' % elem for elem in list(Counter(banco).keys())])))
    
    def percorre_arvore(self, linha, no_pai):

        if type(no_pai) == Folha:
            return no_pai.classe

        else:
            indice_pergunta = list(self.banco_de_dados.columns).index(no_pai.pergunta.coluna)

            if linha[indice_pergunta] >= no_pai.pergunta.valor:
                return self.percorre_arvore(linha, no_pai.filho_direito)

            else:
                return self.percorre_arvore(linha, no_pai.filho_esquerdo)

    def classifica(self, banco):

        serie_predicao = []

        for index, linha in banco.iterrows():
            classe = self.percorre_arvore(linha, self.raiz)
            serie_predicao.append(classe)

        serie_predicao = pd.Series(serie_predicao, name='predicao')
        predicao = pd.concat([banco[self.coluna_alvo], serie_predicao], axis=1)

        return predicao


class No(object):
    
    def __init__(self, nome=None, entropia=None, pergunta=None, filhos=[None, None]):
        
        self.nome = nome
        self.entropia = entropia
        self.pergunta = pergunta
        self.filho_esq = filhos[0]
        self.filho_dir = filhos[1]
    
    def __repr__(self):
        return '{} - {} - {}'.format(self.nome, self.entropia, self.pergunta)


class Folha(object):
    
    def __init__(self, banco, alvo):
        self.banco = banco
        self.alvo = alvo
        self.filho_esq = None
        self.filho_dir = None
        self.classe = st.mode(self.banco[self.alvo])
    
    def __repr__(self):
        return '{}'.format(self.classe)


class Pergunta(object):

    def __init__(self, coluna, valor):
        self.coluna = coluna
        self.valor = valor

    def numerico(self, valor):
        return isinstance(valor, int) or isinstance(valor, float)

    def verifica(self, exemplo):
        valor = exemplo[self.coluna]

        if self.numerico(valor):
            return valor >= self.valor
        else:
            return valor == self.valor

    def __repr__(self):
        condicao = "=="
        if self.numerico(self.valor):
            condicao = ">="
        return "%s %s %s?" % (self.coluna, condicao, str(self.valor))