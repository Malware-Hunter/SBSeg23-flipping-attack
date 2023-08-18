from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import sys
import pandas as pd
from os.path import basename
from statistics import mean


def generate_df(name_csv_file):
	df = pd.read_csv(name_csv_file)
	return df

def get_X_y(column, dataset):
	if column not in dataset.columns:
		info = colored(f'Class Column {column} Not Found in Dataset {args.dataset}', 'red')
		print(info)
	X = dataset.drop(columns = column)
	y = dataset[column]
	return X, y

def get_sorted(R):
	R_sorted_values = sorted(R.values())
	R_sorted = {}
	for i in R_sorted_values:
		for j in R.keys():
			if R[j] == i:
				R_sorted[j] = R[j]
	return R_sorted

def get_silhouette(dataset, fraction):
	flip_0_to_1 = 0
	flip_1_to_0 = 0
	X, Y = get_X_y('class',dataset)
	#print('X_train dataset')
	#print(X)
	#print('Y_train dataset')
	#print(Y)
	km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=100, random_state=42).fit(X)
	
	labels = km.labels_
	print('Labels: ', labels)
	samples = silhouette_samples(X, labels, metric='euclidean')
	S = {}
	i_samples = 0
	for row in X.index:
		S[row] = samples[i_samples]
		i_samples +=1 

	for row in S.keys():
		if S[row] <= 0:
			if Y[row] > 0:
				flip_1_to_0 += 1
			else:
				flip_0_to_1 += 1
			Y[row] = abs(1 - Y[row])

	#dataset.to_csv(f'teste_silhouette_all.csv', index = False)
	#print(Y)
	#return Y, flip_0_to_1, flip_1_to_0

	#Ordenar S
	S = get_sorted(S)

	list_indexs = list(S.keys())
	list_indexs_to_flip = list_indexs[:round(len(list_indexs)*fraction)]
	print('List size:' ,len(list_indexs_to_flip))
	for item in list_indexs_to_flip:
		dataset.loc[[item],['class']] = abs(1 - Y[item])

	#dataset.to_csv(f'silhouette_{100*fraction}_.csv', index = False)
	X_, Y_flipped = get_X_y('class',dataset)
	return Y_flipped
	

if __name__=="__main__":
    dataset_path = sys.argv[1]
    df = generate_df(dataset_path)
    get_silhouette(df, 0.20)