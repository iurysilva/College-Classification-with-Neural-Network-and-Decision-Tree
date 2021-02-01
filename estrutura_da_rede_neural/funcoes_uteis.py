import numpy as np

#A função sigmoide é a principal função de ativação do algoritmo de rede neural,
#ela será utilizada tanto na etapa de aprendizado quanto nas etapas de teste.
def sigmoide(x):
    return 1/(1+np.exp(-x))

#A derivada da função de ativação é necessária na etapa de propagação do erro,
#ou seja, na função Backpropagation.
def derivar_sigmoide(x):
    return x * (1 - x)

#A derivada da função de ativação é necessária na etapa de propagação do erro,
#ou seja, na função Backpropagation.
def tahn(x):
    return 2 / (1 + np.exp(-2*x)) - 1

#A função multiplica matrizes simplesmente recebe duas matrizes e as multiplica
#utilizando um método da biblioteca Numpy.
def multiplicar_matrizes(a, b):
    return np.matmul(a, b)
