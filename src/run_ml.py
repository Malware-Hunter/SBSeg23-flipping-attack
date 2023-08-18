from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import math
import sys
import os
import argparse
import numpy as np
import pandas as pd
import timeit
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from os.path import basename, dirname, exists
from flip_random_method import random_method
from entropy_method import entropy_method
from flip_silhouette_method import get_silhouette
elapsed_time = 0

def parse_args(argv):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '-d', '--dataset', metavar='DATASET',
        help='Dataset (csv file).', type=str, required=True)
    parser.add_argument(
        '-f', '--decimal_fraction', metavar='DECIMAL_FRACTION',
        help='Decimal fraction of flip between 0 and 1.', type=float, required=True)
    parser.add_argument(
        '-c', '--classifier', metavar='CLASSIFIER',
        help="Classifier.",
        choices=['svm', 'rf', 'dt', 'nb'],
        type=str, default='svm')
    parser.add_argument(
        '-n', '--filename', metavar='FILENAME',
        help='filename(output report file).', type=str, required=False)
    args = parser.parse_args(argv)
    return args

if __name__=="__main__":
    args = parse_args(sys.argv[1:])

    try:
        dataset = pd.read_csv(args.dataset)
    except BaseException as e:
        print('Exception: {}'.format(e))
        exit(1)

    class_ = dataset['class']
    skf = StratifiedKFold(n_splits = 5)
    pred = []
    clss = []
    fold = 1
    #list_0_to_1 = []
    #list_1_to_0 = []
    #size_of_train_dataset = []
    #flip_0_to_1 = 0
    #flip_1_to_0 = 0
    for train_index, test_index in skf.split(dataset, class_):
        train = dataset.loc[train_index,:]
        #train = entropy_method(train, args.decimal_fraction, args.dataset) #quando usar o método de entropia
        X_train = train.iloc[:,:-1] # features
        y_train = train.iloc[:,-1] # class
        y_train = random_method(y_train, args.decimal_fraction) #quando usar o método aleatório
        #y_train = get_silhouette(train, args.decimal_fraction) #quando usar o método silhouette clustering
        #size_of_train_dataset.append(len(X_train))
        #list_0_to_1.append(flip_0_to_1)
        #list_1_to_0.append(flip_1_to_0)
        test = dataset.loc[test_index,:]
        X_test = test.iloc[:,:-1] # featuresg
        y_test = test.iloc[:,-1] # class
        if args.classifier == 'svm':
            clf = svm.SVC()
        if args.classifier == 'rf':
            clf = RandomForestClassifier(random_state = 0)
        if args.classifier == 'dt':
            clf = DecisionTreeClassifier()
        if args.classifier == 'nb':
            clf = GaussianNB()
        print('Fold %s - Fit Model' % fold)
        start_time = timeit.default_timer()
        clf.fit(X_train, y_train)
        end_time = timeit.default_timer()
        #print("Elapsed Time:", end_time - start_time)
        print('Fold %s - Predict' % fold)
        #start_time = timeit.default_timer()
        pred_f = clf.predict(X_test)
        end_time = timeit.default_timer()
        #print("Elapsed Time:", end_time - start_time)
        pred.extend(pred_f)
        clss += list(y_test)
        fold += 1
        elapsed_time += end_time - start_time

    tn, fp, fn, tp = confusion_matrix(clss, pred).ravel()
    accuracy = metrics.accuracy_score(clss, pred)
    precision = metrics.precision_score(clss, pred, zero_division = 0)
    recall = metrics.recall_score(clss, pred, zero_division = 0)
    f1_score = metrics.f1_score(clss, pred, zero_division = 0)
    roc_auc = metrics.roc_auc_score(clss, pred)
    precision *= 100.0
    accuracy *= 100.0
    recall *= 100.0
    f1_score *= 100.0
    roc_auc *= 100.0
    mcc = metrics.matthews_corrcoef(clss, pred)
    x = basename(args.dataset)
    data = {
        'Classifier': [args.classifier],
        'Dataset': x,
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1_Score': [f1_score],
        'MCC': [mcc],
        'RoC_AuC': [roc_auc],
        'TP': [tp],
        'TN': [tn],
        'FP': [fp],
        'FN': [fn],
        'Elapsed Time': [elapsed_time],
    }
    df = pd.DataFrame(data)
    csv_filename = f'{args.filename}.csv'
    file_exists = exists(csv_filename)
    df.to_csv(csv_filename, mode='a', index = False, header = not file_exists)
    print(df)
