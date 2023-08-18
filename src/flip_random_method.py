import pandas as pd
import random
import sys
from os.path import basename

def generate_indexs(Y_train):
	list_indexs = []
	for i in Y_train.index:
		if Y_train.get(i) == 1:
			list_indexs.append(i)
	return list_indexs

def generate_indexs_to_flip(l,p):
	list_indexs_to_flip = []
	amount = round(len(l)*p)
	for i in range(amount):
		item = random.choice(l)
		list_indexs_to_flip.append(item)
		l.remove(item)
	return list_indexs_to_flip


def random_method(Y_train, percent):
	print('Percentage: ', percent*100)
	print('Malwares in original dataset:', Y_train.sum()) #testes
	list_indexs = generate_indexs(Y_train)
	list_indexs_to_flip = generate_indexs_to_flip(list_indexs, percent)
	for i in list_indexs_to_flip:
		Y_train.update(pd.Series([0], index = [i]))
	print('Malwares in flipped dataset', Y_train.sum())
	return Y_train
	


