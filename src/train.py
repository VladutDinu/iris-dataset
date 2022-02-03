import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import sys
import pickle
import yaml
data_path = sys.argv[1]
model_path = sys.argv[2]
params = yaml.safe_load(open('params.yml'))
split = params['prepare']['split']
knn = KNeighborsClassifier(n_neighbors=params['features']['kN'])

df = pd.read_csv(data_path).drop(['index'], axis=1)

X = df.drop(['y'], axis=1)
y = df['y']

knn.fit(X[:split], y[:split])

# Its important to use binary mode 
knnPickle = open(model_path, 'wb') 

# source, destination 
pickle.dump(knn, knnPickle)     

#y_pred = knn.predict(X[130:])
#print(metrics.accuracy_score(y[130:], y_pred))