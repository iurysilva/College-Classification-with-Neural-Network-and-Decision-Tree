import math
import pandas as pd
import numpy as np
from collections import Counter
import statistics as st


# Classe englobando a estrutura da Árvore de Decisão e todas as suas funções
class ArvoreDecisao(object):

    # A classe recebe o banco de dados e o alvo
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

    # Função que mede a altura da árvore, chama uma função privada
    def altura(self):
        return self._altura_recursiva(self.raiz, 0)

    # Aqui a árvore é percorrida de forma recursiva, e o valor da maior altura encontrada é retornada.
    def _altura_recursiva(self, no_atual, altura_atual):
        if not no_atual:
            return altura_atual
        altura_esquerdo = self._altura_recursiva(no_atual.filho_esquerdo, altura_atual + 1)
        altura_direito = self._altura_recursiva(no_atual.filho_direito, altura_atual + 1)
        return max(altura_esquerdo, altura_direito)

    # Função responsável pela criação da árvore, chamada quando o objeto é instanciado.
    def cria_arvore(self):
        self.raiz = self._cria_arvore_recursiva(self.banco_de_dados)

    # Cria a árvore de forma recursiva, recebendo apenas o banco de dados.
    def _cria_arvore_recursiva(self, banco):

        # O primeiro nó é criado, chamando a função responsável por achar o melhor corte
        no = self.verifica_melhor_corte(banco)

        # Aqui calculamos o número de ocorrência de cada classe no subconjunto "esquerdo" do banco.
        n_ocorrencias_classes_esquerda = list(Counter(no.filho_esquerdo[self.coluna_alvo]).values())
        n_classes_esquerda = len(list(Counter(no.filho_esquerdo[self.coluna_alvo]).values()))

        # Aqui calculamos o número de ocorrência de cada classe no subconjunto "direito" do banco.
        n_ocorrencias_classes_direita = list(Counter(no.filho_direito[self.coluna_alvo]).values())
        n_classes_direita = len(list(Counter(no.filho_direito[self.coluna_alvo]).values()))

        dominancia_esquerda = 0
        dominancia_direita = 0

        # Aqui é feito o cálculo da classe dominante do subconjunto "esquerdo" do banco.
        for index_classe in range(n_classes_esquerda):
            dominancia_atual = n_ocorrencias_classes_esquerda[index_classe] / sum(n_ocorrencias_classes_esquerda)
            if dominancia_atual > dominancia_esquerda:
                dominancia_esquerda = dominancia_atual

        # Aqui é feito o cálculo da classe dominante do subconjunto "direito" do banco.
        for index_classe in range(n_classes_direita):
            dominancia_atual = n_ocorrencias_classes_direita[index_classe] / sum(n_ocorrencias_classes_direita)
            if dominancia_atual > dominancia_direita:
                dominancia_direita = dominancia_atual

        # Como condição de parada, optamos por uma taxa de 80% de dominância de uma
        # mesma classe no subconjunto do banco, portanto, se isso ocorrer,
        # este subconjunto torna-se um nó folha. Senão, o algoritmo continua
        # e a árvore se divide novamente.
        if dominancia_esquerda < 0.8:
            no.filho_esquerdo = self._cria_arvore_recursiva(no.filho_esquerdo)
        else:
            no.filho_esquerdo = Folha(no.filho_esquerdo, self.coluna_alvo)

        if dominancia_direita < 0.8:
            no.filho_direito = self._cria_arvore_recursiva(no.filho_direito)
        else:
            no.filho_direito = Folha(no.filho_direito, self.coluna_alvo)
            
        return no

    # Neste função verificamos o melhor particionamento do banco,
    # aquele que irá gerar o maior ganho de informação.
    def verifica_melhor_corte(self, banco):
        
        maior_ganho = {'Atributo': '', 'Ganho de Informação': 0, 'Pergunta': None}
        filho_esquerdo = None
        filho_direito = None

        # Cálculo da entropia geral de toda a base de dados.
        entropia_pai = self.calcula_entropia(banco)

        # Para cada atributo e cada valor, dividimos a base de dados em dois,
        # calculamos os ganhos de informação e os armazenamos se for melhor que o anterior.
        for coluna in self.colunas[1:]:

            # Arredondamos os valores, para otimizar o algoritmo.
            banco_coluna = self.arredonda_float(banco[coluna])

            for linha in banco_coluna:
                # Instanciamos a pergunta, de acordo com a divisão atual a ser analisada.
                pergunta = Pergunta(coluna, linha)
                # Temos a base divida pela função, usando a pergunta como parâmetro
                banco_esquerdo, banco_direito = self.corta_banco(banco, pergunta)
                # O ganho de informação é calculado entre os dois subconjuntos de dados
                ganho_info = self.calcula_ganho_informacao(banco_esquerdo, banco_direito, entropia_pai)
                # Se o ganho for melhor que o atual, é armazenado no dicionário
                if ganho_info > maior_ganho['Ganho de Informação']:
                    maior_ganho = {'Atributo': coluna, 'Ganho de Informação': ganho_info, 'Pergunta': pergunta}
                    filho_esquerdo, filho_direito = banco_esquerdo, banco_direito

        # Finalmente, retornamos um nó com todas as informações adquiridas,
        # relacionadas ao maior ganho de informação possível.
        return No(atributo=maior_ganho['Atributo'],
                  entropia=entropia_pai,
                  pergunta=maior_ganho['Pergunta'],
                  filho_esquerdo=filho_esquerdo,
                  filho_direito=filho_direito)

    # Função responsável pelo cálculo de entropia, de acordo com as fórmulas conhecidas.
    def calcula_entropia(self, banco):
        
        entropia = 0
        
        numero_ocorrencias_classe = list(Counter(banco[self.coluna_alvo]).values())

        n_linhas = banco.shape[0]
        
        for quantidade in numero_ocorrencias_classe:
            peso = (quantidade / n_linhas)
            entropia += - peso * math.log(peso, 2)
            
        return entropia

    # Similarmente, é calculado o ganho de informação utilizando os dois subconjuntos
    # da base de dados.
    def calcula_ganho_informacao(self, banco_esquerdo, banco_direito, entropia_pai):
        
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

    # Função responsável por particionar a base de dados de acordo com a
    # pergunta inserida.
    def corta_banco(self, banco, pergunta):

        # Os dois subconjuntos gerados, cada um com as linhas
        # que deram True para a pergunta, ou False.
        linha_true, linha_false = [], []

        # Iteramos a base de dados e cada linha é avaliada, depois
        # adicionada à lista correspondente.
        for _, row in banco.iterrows():

            if pergunta.verifica(row):
                linha_true.append(row)

            else:
                linha_false.append(row)

        # Conversão das listas de volta para o formato da biblioteca Pandas.
        banco_esquerdo = pd.DataFrame(linha_false)
        banco_direito = pd.DataFrame(linha_true)
        
        return banco_esquerdo, banco_direito

    # Função responsável por arredondar os valores possíveis do banco de dados,
    # no momento da partição, a fim de melhorar o tempo de execução do algoritmo.
    def arredonda_float(self, banco):
        return list(map(int, Counter(['%d' % elem for elem in list(Counter(banco).keys())])))
    
    # Função responsável por percorrer a árvore com o indivíduo a ser classificado.
    # Primeiramente, verifica-se o tipo de nó atual, e se for uma folha, a classe é
    # retornada. Senão, a função entra na recursão percorrendo os nós de acordo com
    # as perguntas de cada um relação ao indivíduo.
    def percorre_arvore(self, linha, no_pai):

        if isinstance(no_pai, Folha):
            return no_pai.classe

        else:
            indice_pergunta = list(self.banco_de_dados.columns).index(no_pai.pergunta.coluna)

            if linha[indice_pergunta] >= no_pai.pergunta.valor:
                return self.percorre_arvore(linha, no_pai.filho_direito)

            else:
                return self.percorre_arvore(linha, no_pai.filho_esquerdo)

    # Função responsável por fazer a classificação do banco inserido,
    # após a construção da árvore.
    def classifica(self, banco):

        serie_predicao = []

        # O banco é iterado e cada linha percorre a árvore
        # a fim de ser classificada, e tal resutaldo é armazenado
        # em uma lista.
        for _, linha in banco.iterrows():
            classe = self.percorre_arvore(linha, self.raiz)
            serie_predicao.append(classe)

        # Após esse processo, a lista é convertida para o objeto Series
        # do Pandas e então concatenado com a coluna alvo do banco de dados.
        serie_predicao = pd.Series(serie_predicao, name='predicao')
        predicao = pd.concat([banco[self.coluna_alvo], serie_predicao], axis=1)

        # Retornamos esse Dataframe contendo a coluna alvo e suas predições
        return predicao

    # Função responsável por imprimir a árvore, percorre a estrutura de forma
    # similar a outras funções já apresentadas, de forma recursiva.
    def imprime(self, no_pai, espacamento=""):

        # Se o nó for folha, imprimimos a sua predição, de acordo com a pergunta.
        if isinstance(no_pai, Folha):
            print(f'{espacamento} Predição: {no_pai.classe}')
            return

        print(f'{espacamento} {str(no_pai.pergunta)}')

        # Imprimimos True, ilustrando as respostas à pergunta anterior
        print(f'{espacamento} --> True:')
        self.imprime(no_pai.filho_direito, espacamento + "  ")

        # Imprimimos False, ilustrando as respostas à pergunta anterior
        print(f'{espacamento} --> False:')
        self.imprime(no_pai.filho_esquerdo, espacamento + "  ")

# Construímos duas classes de Nós, uma sendo um Nó padrão, possuindo diversps atributos,
# tendo como destaque o atributo da pergunta, que será utilizada para dividir a base de dados
# em dois novos nós. Este tipo de nó é responsável por tomar a decisão no momento de classificar algum dado.
class No(object):
    
    def __init__(self, atributo=None, entropia=None, pergunta=None, filho_esquerdo=None, filho_direito=None):
        
        self.atributo = atributo
        self.entropia = entropia
        self.pergunta = pergunta
        self.filho_esquerdo = filho_esquerdo
        self.filho_direito = filho_direito
    
    def __repr__(self):
        return '{} - {} - {}'.format(self.atributo, self.entropia, self.pergunta)

# Apesar de não ser necessário para o nó ou a classificação em si, o Nó Folha possui atributos de filhos,
# que são utilizados pela função responsável por calcular a altura da árvore.
class Folha(object):
    
    def __init__(self, banco, alvo):
        self.banco = banco
        self.alvo = alvo
        self.filho_esquerdo = None
        self.filho_direito = None
        self.classe = st.mode(self.banco[self.alvo])
    
    def __repr__(self):
        return '{}'.format(self.classe)

# A classe pergunta é a responsável por dividir o banco de dados a cada nó de decisão, ela armazena
# o atributo e o valor feitos na divisão, como por exemplo "Color Intensity >= 4". Ela é capaz de
# receber até mesmo classes categóricas, embora o banco utilizado no trabalho tenha somente valores númericos.
class Pergunta(object):

    def __init__(self, coluna, valor):
        self.coluna = coluna
        self.valor = valor

    def numerico(self, valor):
        return isinstance(valor, int) or isinstance(valor, float)

    # Esta função nos retorna o valor verdade da pergunta do nó de decisão.
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
