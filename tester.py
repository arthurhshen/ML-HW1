import sys
import scipy.io
import numpy as np
import time
from kNNClass import kNN
from MVGClass import MVG


def main():
    mat_file = sys.path[0] + "/hw1data.mat"
    mat_data = scipy.io.loadmat(mat_file)
    N = len(mat_data['X'])

    X = mat_data['X']
    Y = mat_data['Y']

    # Simple example to classify given a classifier clf
    t0 = time.time()
    X_training_data = X[0:8000, :]
    Y_training_data = Y[0:8000, 0]
    # clf = MVG(train_X, train_Y, max_features=250)
    kNN_classifier = kNN(X_training_data, Y_training_data, metric='L2')
    test = kNN_classifier.classify(X[9001])

    t1 = time.time()
    print("We got: {}, the correct answer was {}. It took {} seconds".format(test, Y[9001][0], t1 - t0))

    t0 = time.time()
    MVG_classifier = MVG(X_training_data, Y_training_data)
    test = MVG_classifier.classify(X[9001])
    t1 = time.time()

    print("We got: {}, the correct answer was {}. It took {} seconds".format(test, Y[9001][0], t1 - t0))


if __name__ == '__main__':
    main()
