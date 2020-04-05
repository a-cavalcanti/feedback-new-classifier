import pandas as pd

import random
##RECEBE O NOME DO CSV
# RETORNA OS DADOS SEM ALTERAÇÃO, DADOS SEM AS CLASSES PARA CLASSIFICAÇÃO E AS CLASSES
import numpy as np
def carregar_dados(csv):

    data = pd.read_csv(csv)
    features = data.columns.difference(['Class'])

    classe = data.keys()[-1] #Pega o nome da coluna das classes

    y = data[classe] #y Recebe as classes de todas as instâncias da base de dados

    for i in data:
        i = str(i)
    X = data.copy()
    del X[classe] #Recebe os demais atributos

    return data, X.values.tolist(), y.tolist(),features


def cruzamento_bases(csv1, csv2):

    atributos_base1, atributos_base2 =  [i for i in pd.read_csv(csv1).keys()], [i for i in pd.read_csv(csv2).keys()]
    intercecao = [i for i in atributos_base1 if i in atributos_base2]
    mapeamento = open("mapeamento").read().split("\n")
    dic = {}
    for i in mapeamento:
        i = i.split(":")
        if i[1]!= '?':
            dic.update({i[1]:i[0]})

    for i in intercecao:
        dic.update({i:i})

    remove1 = [i for i in atributos_base1 if i not in dic.values()]
    remove2 = [i for i in atributos_base2 if i not in dic.keys()]
    return dic, remove1, remove2

def gerando_base(dic, remove1, remove2):
    base1,_,_ = carregar_dados("cognitivePortuguese.csv")
    base2,_,_ = carregar_dados("cognitiveEnglish.csv")

    base2 = base2.rename(columns=dic)

    for i in remove1:
        del base1[i]
    for i in remove2:
        del base2[i]

    base1 = base1.reindex(sorted(base1.columns), axis=1)
    base2 = base2.reindex(sorted(base2.columns), axis=1)
    del base1['Future tense']
    del base1['adverbs' \
              '' \
              '']
    print(base1.keys())
    print(base2.keys())
    print(len(base1.keys()))
    print(len(base2.keys()))
    input()
 #   base1.to_csv("pt")
 #   base2.to_csv("en")

    #print(base2.head())
#dic, r1, r2 = cruzamento_bases("cognitivePortuguese.csv", "cognitiveEnglish.csv")
#for i in dic.values():
 #   print(i)
#gerando_base(dic, r1,r2)