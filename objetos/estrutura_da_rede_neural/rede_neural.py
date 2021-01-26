import numpy as np
from objetos.estrutura_da_rede_neural import Camada


class Rede_Neural:
    def __init__(self, num_entradas, num_saidas,  num_camadas, neuronios_por_camada_oculta, banco):
        self.banco = banco
        self.num_entradas = num_entradas
        self.entradas = np.zeros((num_entradas, 1), dtype="float64")
        self.num_saidas = num_saidas
        self.numero_camadas = num_camadas
        self.neuronios_por_camada_oculta = neuronios_por_camada_oculta
        self.camadas = self.cria_camadas()

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
                linhas = camada_1.numero_neuronios
                colunas = camada_2.numero_neuronios
                sinapses = np.random.rand(linhas, colunas)
                self.camadas[camada].sinapses = np.copy(sinapses)

    def feedfoward(self):
        a = 2

    def backpropagation(self):
        b = 3

    def aprender(self, quantidade_de_linhas_para_ler):
        for linha in range(quantidade_de_linhas_para_ler):
            print(linha)