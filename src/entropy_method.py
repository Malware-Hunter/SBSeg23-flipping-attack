import numpy as np
import pandas as pd
from termcolor import colored, cprint
from tqdm import tqdm
from sklearn_extra.cluster import KMedoids
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from os.path import basename

def get_X_y(column, dataset):
	if column not in dataset.columns:
		info = colored(f'Class Column {column} Not Found in Dataset {args.dataset}', 'red')
		print(info)
	X = dataset.drop(columns = column)
	y = dataset[column]
	return X, y


def get_Ej(S, j, P):
	n = len(S)
	k = 1/np.log(n)
	sum = 0
	for i in S.index:
		pij = get_P_i_j(S, P, i, j)
		sum += 0 if pij == 0 else pij*np.log(pij)
	return -k*sum

def get_P_i_j(S, P, i, j):
	return S._get_value(i, j)*P[j]

def get_sorted(R):
	R_sorted_values = sorted(R.values())
	R_sorted = {}
	for i in R_sorted_values:
		for j in R.keys():
			if R[j] == i:
				R_sorted[j] = R[j]
	return R_sorted


def entropy_method(train_dataset, fraction, dt):
	X_train, y_train = get_X_y('class', train_dataset)
	print(colored('Dataset Before Flip:', 'blue'))
	print(y_train.value_counts())
	malwares_dataset = train_dataset[train_dataset['class'] == 1]

	
	X_malwares, y_malwares = get_X_y('class', malwares_dataset)
	#print('Malware dataset \n', X_malwares)
	P = {}
	for j in X_malwares.columns.values:
		P[j] = 1/X_malwares[j].sum() if X_malwares[j].sum() != 0 else 0


	E = {}
	for j in X_malwares.columns.values:
		E[j] = get_Ej(X_malwares, j, P)

	E_sum = sum(E.values())
	W = {}

	for j in X_malwares.columns.values:
		W[j] = (1-E[j])/(len(E)-E_sum)
	
	R = {}
	for i in X_malwares.index:
		sum_ = 0
		for j in X_malwares.columns.values:
			sum_ += W[j] * get_P_i_j(X_malwares, P, i, j)
		R[i] = sum_
	
	list_indexs = list(get_sorted(R).keys())
	list_indexs_to_flip = list_indexs[:round(len(list_indexs)*fraction)]
	print('List size:' ,len(list_indexs_to_flip))
	for item in list_indexs_to_flip:
		train_dataset.loc[[item],['class']] = 0
	print(colored(f'Dataset After Flip ({100*fraction}%):', 'blue'))
	print(y_train.value_counts())

	#train_dataset.to_csv(f'entropy_{100*fraction}_{basename(dt)}', index = False)
	return train_dataset
	#print(train_dataset)