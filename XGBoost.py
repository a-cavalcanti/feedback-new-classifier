import pandas as pd
import numpy as np
import os
from sklearn.model_selection import RepeatedKFold, cross_validate, RepeatedStratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report
#, mean_squared_error
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import xgboost as xgb
# import nltk
# nltk.download('punkt')
import string
# nlp = spacy.load('pt')


def read_classes(path):
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    print(CURR_DIR)
    data = pd.read_csv(CURR_DIR+'/'+path)
    id_name = data.keys()[0]
    vector = []
    print(id_name)
    # Remove id col
    del data[id_name]
    # 11 -> classes number
    for i in range(11):
        vector.append(data[data.keys()[i]].values.tolist())
    return vector


def read_data(csv):
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))
    print(CURR_DIR)
    data = pd.read_csv(CURR_DIR+'/'+csv)
    # id col name
    id_name = data.keys()[0]
    new_data = data.copy()

    # Remove id col
    # del new_data[id_name]
    return data, new_data.values.tolist()


def error_by_class(confusionmatrix):
    # calculate the error by class using the confusion matrix
    tam = confusionmatrix.shape[0]
    vector = []
    for i in range(tam):
        match = confusionmatrix[i][i]
        total = sum(confusionmatrix[i])
        error_value = round(1 - (match/total), 4)
        vector.append(error_value)
    return vector


def cross_validation(data_x, data_y, k, ntree, results):

    accuracy = []
    kappa = []

    # cross-validation
    rkf = RepeatedStratifiedKFold(n_splits=k, n_repeats=1, random_state=54321)
    classifier = xgb.XGBClassifier(n_estimators=ntree, use_label_encoder=False, eval_metric='mlogloss')
    conf_matrix = np.zeros((2, 2))

    for train_index, test_index in rkf.split(data_x, data_y):
        x_train, x_test = [data_x[i] for i in train_index], [data_x[j] for j in test_index]
        y_train, y_test = [data_y[i] for i in train_index], [data_y[j] for j in test_index]

        x_train_np = np.asarray(x_train)
        x_test_np = np.asarray(x_test)
        y_train_np = np.asarray(y_train)
        y_test_np = np.asarray(y_test)

        # classifier training
        classifier.fit(x_train_np, y_train_np)
        class_predicted = classifier.predict(x_test_np)
        # transform for np
        class_predicted_np = np.asarray(class_predicted)

        accuracy.append(accuracy_score(class_predicted_np, y_test_np))
        kappa.append(cohen_kappa_score(class_predicted_np, y_test_np))

        # sum the confusion matrix for each k-fold execution
        conf_matrix = conf_matrix + confusion_matrix(y_pred=class_predicted_np, y_true=y_test_np)

    # calculating mean and standard deviation for accuracy and kappa
    acc_media = float(np.mean(accuracy))
    acc_std = float(np.std(accuracy))
    kappa_media = float(np.mean(kappa))
    kappa_std = float(np.std(kappa))

    results['ntree'].append(classifier.n_estimators)
    results['error-by-class'].append(error_by_class(conf_matrix))
    results["accuracy-std"].append(str(round(acc_media, 4)) + "(" + str(round(acc_std, 4)) + ")")
    results["accuracy"].append(round(acc_media, 4))
    results["error"].append(round(1 - acc_media, 4))
    results["kappa-std"].append(str(round(kappa_media, 4)) + "(" + str(round(kappa_std, 4)) + ")")
    results["kappa"].append(round(kappa_media, 4))

    return results, classifier


def experiments():

    csv_classes = 'classes.csv'
    # one list with 11 lists
    classes = read_classes(csv_classes)

    csv_features = 'features.csv'
    data_train, features = read_data(csv_features)

    indices = open('features.csv', 'r', encoding='utf-8', errors='ignore').read().split('\n')
    features_list = indices[0].split(',')
    del features_list[0]
    print("Quantity: ")
    print(len(features_list))

    for z in range(len(classes)):

        y = classes[z]
        x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=123)

        j = 0
        best = 0.0
        best_mtree = 0

        results = {}
        results.update({'ntree': []})
        results.update({'accuracy-std': []})
        results.update({'kappa-std': []})
        results.update({'kappa': []})
        results.update({'accuracy': []})
        results.update({'error': []})
        results.update({'error-by-class': []})

        # choosing the best ntree
        for i in range(100, 601, 50):

            results, classifier = cross_validation(x_train, y_train, k=10, ntree=i, results=results)
            print(results["accuracy"])
            if results["accuracy"][j] > best:
                best = results["accuracy"][j]
                best_classifier = classifier
                best_mtree = i
            j = j+1

        print("TRAIN RESULTS: ")
        print(results)
        output_result = open('result-class-' + str(z), 'w')
        output_result.write(str(results))
        output_result.write('\n')

        x_test_np = np.asarray(x_test)
        y_test_np = np.asarray(y_test)

        y_pred = best_classifier.predict(x_test_np)
        y_pred_np = np.asarray(y_pred)
        accuracy = accuracy_score(y_pred_np, y_test_np)
        kappa = cohen_kappa_score(y_pred_np, y_test_np)
        class_report = classification_report(y_test_np, y_pred_np)
        conf_matrix = confusion_matrix(y_pred=y_pred_np, y_true=y_test_np)

        print("TEST RESULTS: ")
        print(class_report)
        print(conf_matrix)
        print("accuracy = ", accuracy)
        print("kappa = ", kappa)

        output_result.write("accuracy = " + str(accuracy) + "\n")
        output_result.write("kappa = " + str(kappa) + "\n")
        output_result.write(str(conf_matrix) + "\n")
        output_result.write(str(class_report) + "\n")
        output_result.close()

        classifier = xgb.XGBClassifier(n_estimators=best_mtree, use_label_encoder=False, eval_metric='mlogloss')

        features_np = np.asarray(features)
        y_np = np.asarray(y)
        classifier.fit(features_np, y_np)

        print("Feature Importance SMOTE" + " class: " + str(z))
        features_importance = zip(classifier.feature_importances_, features_list)
        output = open('feature-importance-classe-'+str(z), 'w')
        for importance, feature in sorted(features_importance, reverse=True):
            print("%s: %f%% - %f%%" % (feature, importance * 100, importance))
            output.write("%s: %f%% - %f%%" % (feature, importance * 100, importance))
            output.write("\n")
        output.close()

        plot_graph(results, z)


def plot_graph(results, z):
    # points on graph
    plt.scatter(results["ntree"], results["error"])
    plt.plot(results["ntree"], results["error"], color='red')
    plt.ylabel("Error rate")
    plt.xlabel("Number of estimators")
    plt.title("Estimator Parameter Tuning ")
    # plt.show()
    figure_name = "class-" + str(z)
    plt.savefig("results-" + figure_name + ".png", format='png')
    plt.close()


def print_results(vector):
    for element in vector:
        print(element)


experiments()
