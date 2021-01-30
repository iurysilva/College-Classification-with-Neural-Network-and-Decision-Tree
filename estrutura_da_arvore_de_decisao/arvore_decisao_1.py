from estrutura_da_arvore_de_decisao.arvore_decisao import Pergunta
import math
import pandas as pd
import numpy as np
from collections import Counter

class No(object):
    def __init__(self, nome=None, entropia=None, pergunta=None, filhos=[None,None], folha=False):
        
        self.nome = nome
        self.entropia = entropia
        self.pergunta = pergunta
        self.filho_esq = filhos[0]
        self.filho_dir = filhos[1]
        self.folha = folha
    
    def __repr__(self):
        return ('{} - {} - {}'.format(self.nome,self.entropia,self.pergunta))


class Arvore_Decisao(object):
        
    def __init__(self, banco_de_dados, coluna_alvo):
        
        self.banco_de_dados = banco_de_dados
        
        self.n_colunas = banco_de_dados.shape[1]
        self.colunas = banco_de_dados.columns
        self.n_linhas = banco_de_dados.shape[0]

        self.coluna_alvo = coluna_alvo
        self.dados_alvo = self.banco_de_dados[coluna_alvo]
        
        self.raiz = None
        
        self.cria_arvore()
    
    
    def __repr__(self):
        return ('Linhas:{}\nColunas:{}'.format(self.n_linhas,self.n_colunas))
    
    
    def cria_arvore(self):
        self.raiz = self.cria_arvore_recursiva(self.banco_de_dados)

    
    def cria_arvore_recursiva(self, banco):
        
        no = self.verifica_melhor_corte(banco)
        flag_esq, flag_dir = False, False
        numero_ocorrencias_classe_esq = Counter(no.filho_esq[self.coluna_alvo]).values()
        numero_ocorrencias_classe_esq = list(numero_ocorrencias_classe_esq)
        
        numero_ocorrencias_classe_dir = Counter(no.filho_dir[self.coluna_alvo]).values()
        numero_ocorrencias_classe_dir = list(numero_ocorrencias_classe_dir)
        
        '''CONDIÇÃO DE PARADA
        '''
        if len(numero_ocorrencias_classe_esq)!=1:
            no.filho_esq=self.cria_arvore_recursiva(no.filho_esq)
            flag_esq = True

            
        if len(numero_ocorrencias_classe_dir) !=1:
            no.filho_dir=self.cria_arvore_recursiva(no.filho_dir)
            flag_dir = True

        if flag_esq and flag_dir:
            no.folha = True

        return no

    
    def verifica_melhor_corte(self, banco):
        
        no = No()
        maior_ganho = ('Coluna', 0)
        filho_esq = None
        filho_dir = None
        
        entropia_pai = self.calcula_entropia(banco)
        
        for coluna in self.colunas[1:]:
            banco_coluna = self.arredonda_float(banco[coluna])
            for linha in banco_coluna[:-1]:
            #for linha in banco[coluna]:
                pergunta = Pergunta(coluna,linha)
                banco_esq, banco_dir = self.corta_banco(banco,pergunta)
                ganho_info = self.calcula_ganho_informacao(banco_esq,banco_dir,entropia_pai)
                if ganho_info > maior_ganho[1]:
                    maior_ganho = (coluna,ganho_info,pergunta)
                    filho_esq, filho_dir = banco_esq, banco_dir
                    
        return No(maior_ganho[0],entropia_pai,maior_ganho[2],[filho_esq,filho_dir])
    

    def calcula_entropia(self,banco):
        
        entropia = 0
        
        numero_ocorrencias_classe = Counter(banco[self.coluna_alvo]).values()
        numero_ocorrencias_classe = list(numero_ocorrencias_classe)
        
        
        numero_linhas = banco.shape[0]
        
        for contagem in numero_ocorrencias_classe:
            peso = (contagem / numero_linhas)
            entropia += - peso * math.log(peso, 2)
            
        return entropia
    
    
    def calcula_ganho_informacao(self, banco_esq, banco_dir, entropia_pai):
        
        ganho_informacao = 0
        
        numero_amostras = banco_esq.shape[0] + banco_dir.shape[0]
        peso_esq = banco_esq.shape[0] / numero_amostras
        peso_dir = banco_dir.shape[0] / numero_amostras
        
        if not banco_esq.empty:
            calculo_esq = peso_esq*self.calcula_entropia(banco_esq)
        else:
            calculo_esq = 0
            
        calculo_dir = peso_dir*self.calcula_entropia(banco_dir)
        
        ganho_informacao = calculo_esq + calculo_dir
        
        return entropia_pai - ganho_informacao
    
    
    def corta_banco(self, banco, pergunta):
        
        linha_true, linha_false = [], []
        
        for _, row in banco.iterrows():
            if pergunta.verifica(row):
                linha_true.append(row)
            else:
                linha_false.append(row)
        
        banco_esq = pd.DataFrame(linha_false)
        banco_dir = pd.DataFrame(linha_true)
        
        return banco_esq, banco_dir
    
    
    def arredonda_float(self, banco):
        return(list(map(int,Counter([ '%d' % elem for elem in list(Counter(banco).keys())]))))
    
    
    def monta_árvore(self):
        pass


class Pergunta(object):

    def __init__(self, coluna, valor):
        self.coluna = coluna
        self.valor = valor
        
        
    def is_numeric(self,valor):
        return isinstance(valor, int) or isinstance(valor, float)
        
        
    def verifica(self, exemplo):
        valor = exemplo[self.coluna]

        if self.is_numeric(valor):
            return valor >= self.valor
        else:
            return valor == self.valor


    def __repr__(self):
        condition = "=="
        if self.is_numeric(self.valor):
            condition = ">="
        return "%s %s %s?" % (
            self.coluna, condition, str(self.valor))