import pandas as pd
import matplotlib.pyplot as plt

def tabela(csv): #RETORNA A TABELA NO FORMATO DO LATEX
    resultados = pd.read_csv(csv)

    mtry = [str(i) for i in resultados['mtry']]
    acuracia = [str(i) for i in resultados['acurácia']]
    kappa = [str(i) for i in resultados['kappa']]

    tabela = open("tabela_pt","w")
    string = ""
    for i in range(4,len(mtry),5):
        string = string + mtry[i] + " & " + acuracia[i] + " & " + kappa[i] + "\\\ \n \hline\n"
    print(string)


def grafico(csv):

    coluna_erro = "erro_classe_"  ##classe
    coluna_ntree = "ntree"  # x
    resultados = pd.read_csv(csv)

    ntree = [int(i) for i in resultados['ntree']]
    erros = []

    for i in range(5):
        erros.append([float(j) for j in resultados[coluna_erro + str(i)]])

    erros.append([float(i.split('(')[0]) for i in resultados['OOB_erro']])

    labelss = ['Classe 0', 'Classe 1', 'Classe 2', 'Classe 3', 'Classe 4', 'OOB']
    linestyles = ['--', '--', '-.', ':', ':','-']
    linewidths = [2, 2, 2, 2, 2,3]
    colors = []
    print(erros)
    for i in range(len(erros)):
        print(i)
        plt.plot(ntree, erros[i], linestyle=linestyles[i], linewidth=linewidths[i], label=labelss[i])

    # plt.set_xdata(ntree)
    plt.legend()
    plt.show()


grafico("resultados_en_pt.csv")


