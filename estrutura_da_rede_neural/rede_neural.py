import numpy as np
from estrutura_da_rede_neural import Camada
from estrutura_da_rede_neural.funcoes_de_ativacao import sigmoide

class Rede_Neural:
    def __init__(self, atributos_entradas, atributos_saidas,  num_camadas, neuronios_por_camada_oculta, banco):
        self.atributos_de_entrada = atributos_entradas
        self.atributos_de_saida = atributos_saidas
        self.banco = banco
        self.num_entradas = self.atributos_de_entrada.size
        self.num_saidas = self.atributos_de_saida.size
        self.numero_camadas = num_camadas
        self.neuronios_por_camada_oculta = neuronios_por_camada_oculta
        self.camadas = self.cria_camadas()
        self.valor_esperado = None

    def mostra_informacoes_das_camadas(self):
        print('')
        for camada in range(self.numero_camadas):
            print("Informações da camada %d: " % camada)
            print("Numero de neuronios: ", self.camadas[camada].numero_neuronios)
            print("Camada final?: ", self.camadas[camada].final)
            print("Sinapses da camada: ")
            print(self.camadas[camada].sinapses, "\n")

    def cria_camadas(self):
        camadas = []
        for camada in range(self.numero_camadas):
            if camada == 0:
                camadas.append(Camada(self.num_entradas))
            elif camada == self.numero_camadas-1:
                camadas.append(Camada(self.num_saidas))
            else:
                camadas.append(Camada(self.neuronios_por_camada_oculta))
        camadas[-1].final = True
        return camadas

    def insere_sinapses(self):
        for camada in range(self.numero_camadas):
            if not self.camadas[camada].final:
                camada_1 = self.camadas[camada]
                camada_2 = self.camadas[camada+1]
                linhas = camada_2.numero_neuronios
                colunas = camada_1.numero_neuronios
                sinapses = np.random.rand(linhas, colunas)
                self.camadas[camada].sinapses = np.copy(sinapses)

    def inserir_entradas(self, linha, banco):
        for entrada in range(self.num_entradas):
            atributo = self.atributos_de_entrada[entrada]
            self.camadas[0].neuronios[entrada] = sigmoide(banco[linha][atributo])

    def inserir_saidas(self, linha, banco):
        for saida in range(self.num_saidas):
            atributo = self.atributos_de_saida[saida]
            self.valor_esperado = sigmoide(banco[linha][atributo])

    def feedfoward(self):
        for camada_atual in range(self.numero_camadas-1):
            camada = self.camadas[camada_atual]
            if not camada.final:
                multiplicacao_matricial = np.matmul(camada.sinapses, camada.neuronios)
                self.camadas[camada_atual + 1].neuronios = sigmoide(multiplicacao_matricial)
                print('sinapses que ligam a próxima camada: ')
                print(camada.sinapses)
                print('neuronios da camada atual: ')
                print(camada.neuronios)
                print('multiplicação das sinapses pelos neuronios: ')
                print(self.camadas[camada_atual+1].neuronios, '\n')
        print("valor na camada final: ", self.camadas[-1].neuronios)
        print("valor esperado: ", self.valor_esperado)

    def backpropagation(self):
        b = 3

    def aprender(self, quantidade_de_linhas_para_ler):
        banco = self.banco.values
        for linha in range(quantidade_de_linhas_para_ler):
            self.inserir_entradas(linha, banco)
            self.inserir_saidas(linha, banco)
            self.feedfoward()
