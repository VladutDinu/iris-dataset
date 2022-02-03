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
params = yaml.safe_load(open('params.yaml'))
split = params['prepare']['split']

df = pd.read_csv(data_path).drop(['index'], axis=1)

X = df.drop(['y'], axis=1)
y = df['y']

knn = pickle.load(open(model_path, 'rb'))

y_pred = knn.predict(X[split:])
print(metrics.accuracy_score(y[split:], y_pred))