import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, cross_validate, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report#, mean_squared_error
from carregar_dados import carregar_dados
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from collections import Counter
import nltk
#nltk.download('punkt')
import string
# nlp = spacy.load('pt')



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

        resultados["erro_classe_"+str(i)].append(taxa_erro)


def validacao_cruzada(X, y, features, k, ntree, resultados ):
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
    resultados_parciais.update({'ntree': []})
    resultados_parciais.update({'mtry': []})
    resultados_parciais.update({'acurácia': []})
    resultados_parciais.update({'kappa': []})
    resultados_parciais.update({'accuracy': []})
    resultados_parciais.update({'erro': []})

    ## VALIDAÇÃO CRUZADA

    rkf = RepeatedStratifiedKFold(n_splits=k, n_repeats=1, random_state=54321) #DIVIDI OS DADOS NOS CONJUNTOS QUE SERÃO DE      TREINO E TESTE EM CADA RODADA DA VALIDAÇÃO CRUZZADA

    matriz_confusao = np.zeros((2,2))


    for train_index, test_index in rkf.split(X, y):
        X_train, X_test = [X.iloc[i] for i in train_index], [X.iloc[j] for j in test_index]
        y_train, y_test = [y.iloc[i] for i in train_index], [y.iloc[j] for j in test_index]

        X_train_np = np.asarray(X_train)
        X_test_np = np.asarray(X_test)
        y_train_np = np.asarray(y_train)
        y_test_np = np.asarray(y_test)

        classificador = xgb.XGBClassifier(n_estimators=ntree)
        classificador.fit(X_train_np, y_train_np)
        y_pred = classificador.predict(X_test_np)

        resultados_parciais["acurácia"].append(accuracy_score(y_pred, y_test_np))
        resultados_parciais["kappa"].append(cohen_kappa_score(y_pred, y_test_np))

        matriz_confusao = matriz_confusao + confusion_matrix(y_pred=y_pred, y_true=y_test_np) ##A MATRIZ DE CONFUSÃO FINAL SERÁ A SOMA DAS MATRIZES DE CONFUSÃO DE CADA RODADA DO KFOLD


    ## SALVANDO OS PARÊMTROS E RESULTADOS DO EXPERIMENTO


    #print(matriz_confusao)
    resultados['ntree'].append(classificador.n_estimators)
    erro_por_classe(matriz_confusao, resultados)

    media = np.mean(resultados_parciais["acurácia"])
    std = np.std(resultados_parciais["acurácia"])
    resultados["acurácia"].append(str(round(media,4))+"("+str(round(std,4))+")")

    resultados["accuracy"].append(round(media, 4))
    resultados["erro"].append(round(1 - media, 4))

    media = np.mean(resultados_parciais["kappa"])
    std = np.std(resultados_parciais["kappa"])
    resultados["kappa"].append(str(round(media, 4)) + "(" + str(round(std, 4)) + ")")



    return resultados, classificador



def experimentos(banco):

##CARREGAR OS DADOS


    dataset = pd.read_csv(banco)
    features = dataset.columns.difference(['id_feedback','class'])

    print(features)

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    ##USANDO SMOTE
    #X_resampled, y_resampled = SMOTE().fit_resample(X, y)
    #print(sorted(Counter(y_resampled).items()))

    ##SEM USAR SMOTE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    ##USANDO SMOTE
    #X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=123)


    resultados = {}
    resultados.update({'ntree': []})
    resultados.update({'mtry': []})
    resultados.update({'acurácia': []})
    resultados.update({'kappa': []})
    resultados.update({'accuracy': []})
    resultados.update({'erro': []})


    classes = set(y_train)
    for i in classes:
        resultados.update({"erro_classe_"+str(i):[]})

    j = 0
    maior = 0.0
    best_mtree = 0

    ##NTREE - TREINAMENTO
    for i in range(100,601,50):
        #mtry = randint(5, 116)

        resultados, classificador = validacao_cruzada(X_train, y_train, features, k=10, ntree=i, resultados=resultados)

        if resultados["accuracy"][j] > maior:
            maior = resultados["accuracy"][j]
            best_classifier = classificador
            best_mtree = i
        j = j+1


    print("RESULTADOS: ")
    print(resultados)

    printResults(resultados['acurácia'])
    printResults(resultados['kappa'])

    X_test_np = np.asarray(X_test)
    y_test_np = np.asarray(y_test)

    y_pred = best_classifier.predict(X_test_np)
    accuracy = accuracy_score(y_pred, y_test_np)
    kappa = cohen_kappa_score(y_pred, y_test_np)
    print(classification_report(y_test_np, y_pred))
    # oob = 1 - classificador.oob_score_

    print(confusion_matrix(y_pred=y_pred, y_true=y_test_np))
    print("accuracy = ", accuracy)
    print("kappa = ", kappa)


    #PARA OBTER FEATURE IMPORTANCE
    classifier = xgb.XGBClassifier(n_estimators=best_mtree)
    classifier.fit(X, y)

    #USANDO SMOTE
    #classifier.fit(X_resampled, y_resampled)

    print("Feature Importance SMOTE" + " " + banco)
    features_importance = zip(classifier.feature_importances_, features)
    for importance, feature in sorted(features_importance, reverse=True):
        print("%s: %f%% - %f%%" % (feature, importance * 100, importance))


    plt.scatter(resultados["ntree"], resultados["erro"]) #pontos no grafico
    plt.plot(resultados["ntree"], resultados["erro"], color='red')
    plt.ylabel("Error rate")
    plt.xlabel("Number of estimators")
    plt.title("Estimator Parameter Tuning ")
    #plt.show()
    figure_name = banco.replace(".csv", "")
    plt.savefig("resultados-"+figure_name+".png", format='png')
    plt.close()


def printResults(vector):
    for element in vector:
        print(element)


bancogp1 = "banco-gp1.csv"
bancogp3 = "banco-gp3.csv"
bancogp5 = "banco-gp5.csv"
bancogp6 = "banco-gp6.csv"
bancoft = "banco-lak-FT.csv"
bancofp = "banco-lak-FP.csv"
bancofs = "banco-lak-FS.csv"

experimentos(bancogp1)
experimentos(bancogp3)
experimentos(bancogp5)
experimentos(bancogp6)
experimentos(bancoft)
experimentos(bancofp)
experimentos(bancofs)