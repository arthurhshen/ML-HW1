import sys
import scipy.io
import numpy as np
import time
from kNNClass import kNN


mat_file = sys.path[0] + "/hw1data.mat"
hw1data = scipy.io.loadmat(mat_file)
N = len(hw1data['X'])

X = np.array(hw1data['X'])
Y = np.array(hw1data['Y'])

# Simple example to classify given a classifier clf
tic = time.clock()
train_X = X[0:8000, :]
train_Y = Y[0:8000, 0]
#clf = MVG(train_X, train_Y, max_features=250)
clf = kNN(train_X, train_Y, metric='L2', k=5)
test = clf.classify(X[9559])
print(test)
print(Y[9559])
