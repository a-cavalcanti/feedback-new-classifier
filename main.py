import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score#, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from carregar_dados import carregar_dados
from random import randint
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')
import string
# nlp = spacy.load('pt')


def avaliacao(y_pred, y_test, metricas, classificador, resultados_parciais):
    ## ADICIONA AO DICIONÁRIO DE RESULTADOS DADO COMO PARÊMTRO A AVALIAÇÃO DOS RESULTADOS DE ACORDO COM AS MÉTRICAS
    # **
    # y_pred classes preditas pelo classificador
    # y_test classes corretas
    # classificador do sklearn no qual os dados foram treinados e testados
    # resultados_parciais dicionário de resultados onde cada elemento equivale a uma métrica e contém uma lista do resultado de
    # cada fold
    # **
    # retorna o dicionário resultados_parciais com os resultados da rodada atual da validação cruzada

    for metrica in metricas:
        if metrica == "acurácia":
            accuracy = accuracy_score(y_pred, y_test)
            resultados_parciais[metrica].append(accuracy)
        elif metrica == "kappa":
            resultados_parciais[metrica].append(cohen_kappa_score(y_pred, y_test))
        elif metrica == "OOB_erro":
            resultados_parciais[metrica].append(1 - classificador.oob_score_)

    return accuracy





def erro_por_classe(matriz_confusao, resultados):
    # A PARTIR DA MATRIZ DE CONFUSÃO CALCULA O ERRO POR CLASSE
    # **
    # matriz_confusao - soma das matrize de confusão de cada fold do tipo ndarray(numpy) de forma NxN
    # resultados - dicionário de todos os resultados
    # **
    # retorna o dicionário resultados com os erros de cada classe preenchidos

    tam = matriz_confusao.shape[0]

    for i in range(tam):

        acerto = matriz_confusao[i][i]
        total = sum(matriz_confusao[i])

        taxa_erro = round(1 - (acerto / total),4)
        print(taxa_erro)

        resultados["erro_classe_"+str(i)].append(taxa_erro)







def validacao_cruzada(X, y, X1, y1, k, ntree, mtry, metricas, resultados ):
    ##É REALIZADO OS O EXPERIMENTO COM VALIDAÇÃO CRUZADA E OS RESULTADOS É ADICIONADO A UM DICIONÁRIO
    # **
    # X - dados
    # y - classes
    # k - número de folds
    # ntree - Número de árvores
    # mtry - número de features
    # metricas - lista de metricas que serão utilizadas na avaliacão( "acurácia","kappa", "OOB_erro")
    # resultados - dicionário que vai ser utilizado para cada experimento, salvando os resultados em um dicionário para ser salvo em CSV
    # **
    # retorna o dicionário resultados com os resultados desse experimento adicionados

    resultados_parciais = {} #SALVAR RESULTADOS DE CADA RODADA DA VALIDAÇÃO CRUZADA

    for metrica in metricas:
        resultados_parciais.update({metrica: []})

    ## VALIDAÇÃO CRUZADA

    rkf = RepeatedStratifiedKFold(n_splits=k, n_repeats=1, random_state=54321) #DIVIDI OS DADOS NOS CONJUNTOS QUE SERÃO DE      TREINO E TESTE EM CADA RODADA DA VALIDAÇÃO CRUZZADA

    matriz_confusao = np.zeros((2,2))


    for train_index, test_index in rkf.split(X, y):

        if X1 == None:
            X_train, X_test = [X[i] for i in train_index], [X[j] for j in test_index]
            y_train, y_test = [y[i] for i in train_index], [y[j] for j in test_index]
        else:

            X_train, X_test = [X[i] for i in train_index], [X1[j] for j in test_index]
            y_train, y_test = [y[i] for i in train_index], [y1[j] for j in test_index]

        classificador = RandomForestClassifier(n_estimators=ntree, max_features=mtry, warm_start=True, oob_score=True)
        classificador.fit(X_train,y_train)
        y_pred = classificador.predict(X_test)

        avaliacao(y_pred, y_test, metricas, classificador, resultados_parciais)

        matriz_confusao = matriz_confusao + confusion_matrix(y_pred=y_pred, y_true=y_test) ##A MATRIZ DE CONFUSÃO FINAL SERÁ A SOMA DAS MATRIZES DE CONFUSÃO DE CADA RODADA DO KFOLD

    ## SALVANDO OS PARÊMTROS E RESULTADOS DO EXPERIMENTO

    print(matriz_confusao)
    resultados['ntree'].append(classificador.n_estimators)
    resultados['mtry'].append(classificador.max_features)
    erro_por_classe(matriz_confusao,resultados)

    for metrica in metricas:

        media = np.mean(resultados_parciais[metrica])
        std = np.std(resultados_parciais[metrica])

        resultados[metrica].append(str(round(media,4))+"("+str(round(std,4))+")")

    return resultados, classificador

def search(lista, valor):
    return [lista.index(x) for x in lista if valor in x]

def extractLiwc(text):

    # reading liwc
    wn = open('LIWC2007_Portugues_win.dic.txt', 'r', encoding='ansi', errors='ignore').read().split('\n')
    wordSetLiwc = []
    for line in wn:
        words = line.split('\t')
        if (words != []):
            wordSetLiwc.append(words)

    # indexes of liwc
    indices = open('indices.txt', 'r', encoding='utf-8', errors='ignore').read().split('\n')

    # dataset tokenization
    wordsDataSet = []


    wordsLine = []
    for word in word_tokenize(text):
        if word not in string.punctuation + "\..." and word != '``' and word != '"':
            wordsLine.append(word.lower())


    # initializing liwc with zero
    liwc = [0] * len(indices)
    #liwc.append([0] * len(indices))

    # print(liwc)

    # performing couting

    print("writing liwc ")


    for word in wordsLine:
        position = search(wordSetLiwc, word)
        if position != []:
            tam = len(wordSetLiwc[position[0]])
            for i in range(tam):
                if wordSetLiwc[position[0]][i] in indices:
                    positionIndices = search(indices, wordSetLiwc[position[0]][i])
                    liwc[positionIndices[0]] = liwc[positionIndices[0]] + 1


    return liwc



sentilex = []
cohmetrix = []
liwc = []

#------------------------------------ADITIONAL FEATURES---------------------------------------------------------------#
#aditional features
def aditionals(post):
    postOriginal = post.lower()
    #post = nlp(post)

    greeting = sum([word_tokenize(postOriginal).count(word) for word in ['olá', 'oi', 'como vai', 'tudo bem', 'como está', 'como esta', 'bom dia', 'boa tarde', 'boa noite']])
    compliment = sum([word_tokenize(postOriginal).count(word) for word in ['parabéns', 'parabens', 'excelente', 'fantástico', 'fantastico', 'bom', 'bem', 'muito bom', 'muito bem', 'ótimo', 'otimo', 'incrivel', 'incrível', 'maravilhoso', 'sensacional','irrepreensível', 'irrepreensivel', 'perfeito']])
    #ners = len(post.ents)

    return [greeting, compliment]


def experimentos():

##CARREGAR OS DADOS

    csvTest = "baseComClassesTeste.csv"
    data_test,X_test,y_test = carregar_dados(csvTest)

    csvTrain = "baseComClassesTreino.csv"
    data_train,X_train,y_train = carregar_dados(csvTrain)


    #csv = "en - en.csv"
    #data, X1, y1 = carregar_dados(csv)
    X1 = None
    y1 = None
    metricas = ["acurácia","kappa", "OOB_erro"]

    resultados = {}
    resultados.update({'ntree': []})
    resultados.update({'mtry': []})


    classes = set(y_train)
    for i in classes:
        resultados.update({"erro_classe_"+str(i):[]})
    for metrica in metricas:
        resultados.update({metrica: []})

##NTREE - TREINAMENTO

    # for i in range(100,501,50):
    #     mtry = randint(5, 116)
    #     ntree = i
    #
    #     resultados = cross_validation(X, y, X1, y1, k=10, ntree=ntree,mtry=mtry, metricas=metricas, resultados=resultados)
    #     print(resultados)

    resultados, classificador = validacao_cruzada(X_train, y_train, X1, y1, k=10, ntree=200,mtry=37, metricas=metricas, resultados=resultados)
    print(resultados)
    y_pred = classificador.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    kappa = cohen_kappa_score(y_pred, y_test)
    oob = 1 - classificador.oob_score_
    print(confusion_matrix(y_pred=y_pred, y_true=y_test))

    print(accuracy)
    print(kappa)
    print(oob)

    texto = "Ter cuidado com as cópias fieis da internet. Procurar ler e escrever com suas palavras a partir do que entendeu."
    liwc = extractLiwc(texto)
    adds = aditionals(texto)
    cohmetrix = [50.0,0.0,500.0,86.405,450.0,2.0,2.5,10.0,150.0,1.0,2.0,20.0,100.0,300.0,50.0,0.0,0.0,0.0,50.0,76562.5,3441.5,1.0,0.0,0.0,0.95,0.25,250.0,0.0,150.0,0.0,50.0,0.0,100.0,0.0,0.0,0.0,0.0,0.0,0.0,1.66666666666667,10.8,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    features = []
    for x in liwc:
        features.append(x)
    for y in cohmetrix:
        features.append(y)
    features.append(adds[0])
    features.append(adds[1])
    features.append(0)
    features.append(0)

    newfeatures = []

    newfeatures.append(features)
    y_pred = classificador.predict(newfeatures)
    print("classe predita ", y_pred)


#SALVAR RESULTADOS
    # path = "resultados_pt_en.csv"
    # resultados = pd.DataFrame.from_dict(resultados)
    # resultados.to_csv(path, index=False)
    # print("salvou")


experimentos()
