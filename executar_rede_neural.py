from arquiteturas_rede_neural import criar_arquitetura_1
from arquiteturas_rede_neural import criar_arquitetura_2
from estrutura_da_rede_neural.funcoes_uteis import sigmoide

arquitetura = criar_arquitetura_2()
linhas_para_aprender = 130

rede = arquitetura
rede.insere_sinapses_e_bias()
rede.aprender(linhas_para_aprender)
