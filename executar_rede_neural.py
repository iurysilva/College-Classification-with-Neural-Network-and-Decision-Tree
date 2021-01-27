from arquiteturas_rede_neural import criar_arquitetura_1

arquitetura = criar_arquitetura_1()
linhas_para_aprender = 100

rede = arquitetura
rede.insere_sinapses()
rede.aprender(linhas_para_aprender)