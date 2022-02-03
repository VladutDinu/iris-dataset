import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import sys
iris = datasets.load_iris()
X1 = np.array(iris.data[:, 0])
X2 = np.array(iris.data[:, 1])
X3 = np.array(iris.data[:, 2])
X4 = np.array(iris.data[:, 3])
y = iris.target
path = sys.argv[1]
d = {
    'x1': X1,
    'x2': X2,
    'x3': X3,
    'x4': X4,
    'y': y
}

df = pd.DataFrame(d)
df.index.name = 'index'
df.to_csv(path)
print('DATA PROCESSED AND SAVED')