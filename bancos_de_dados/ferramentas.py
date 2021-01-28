def retorna_treino(base):
    treino = base.sample(frac=0.7,random_state=200)
    return treino

def retorna_teste(base, base_treino):
    teste = base.drop(base_treino.index)
    return teste