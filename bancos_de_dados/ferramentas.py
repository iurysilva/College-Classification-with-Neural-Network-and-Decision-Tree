import numpy as np

def retorna_treino(base):
    treino = base.sample(frac=0.7)
    return treino

def retorna_teste(base, base_treino):
    teste = base.drop(base_treino.index)
    return teste

def calcula_resultados(matriz):
    num_classes = len(matriz)
    sensibilidade = np.zeros([num_classes])
    especificidade = np.zeros([num_classes])
    confiabilidade_positiva = np.zeros([num_classes])
    confiabilidade_negativa = np.zeros([num_classes])
    tp = np.zeros([num_classes])
    tn = np.zeros([num_classes])
    fn = np.zeros([num_classes])
    fp = np.zeros([num_classes])
    acertos = 0
    acuracia = 0
    total = 0

    for classe in range(num_classes):
        for linha in range(num_classes):
            for coluna in range(num_classes):
                if classe == linha == coluna:
                    tp[classe] += matriz[linha][coluna]
                elif classe != linha == coluna:
                    tn[classe] += matriz[linha][coluna]
                elif classe == linha != coluna:
                    fn[classe] += matriz[linha][coluna]
                elif classe == coluna != linha:
                    fp[classe] += matriz[linha][coluna]

    print('TP = ', tp)
    print('TN = ', tn)
    print('FN = ', fn)
    print('FP = ', fp)

    for linha in range(num_classes):
        for coluna in range(num_classes):
            if linha == coluna:
                acertos += matriz[linha][coluna]
            total += matriz[linha][coluna]

    acuracia = (acertos*100)/total

    for classe in range(num_classes):
        sensibilidade[classe] = tp[classe]/(tp[classe] + fn[classe])
        especificidade[classe] = tn[classe]/(tn[classe] + fp[classe])
        confiabilidade_positiva[classe] = tp[classe]/(tp[classe] + fp[classe])
        confiabilidade_negativa[classe] = tn[classe]/(tn[classe] + fn[classe])
        print('----------- Classe %d -----------' %(classe+1))
        print('Sensibilidade: ', sensibilidade[classe])
        print('Especificidade: ', especificidade[classe])
        print('Confiabilidade Positiva: ', confiabilidade_positiva[classe])
        print('Confiabilidade Negativa: ', confiabilidade_negativa[classe])

    print(acuracia)
    return acuracia