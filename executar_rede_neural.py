from arquiteturas_rede_neural import criar_arquitetura_1
from arquiteturas_rede_neural import criar_arquitetura_2
from arquiteturas_rede_neural import criar_arquitetura_3
from arquiteturas_rede_neural import criar_arquitetura_4
from estrutura_da_rede_neural.funcoes_uteis import sigmoide

arquitetura = criar_arquitetura_4()
iteracoes = 10000

rede = arquitetura
rede.learning_rate = 0.5
rede.insere_sinapses_e_bias()
rede.aprender(iteracoes)
