import sys
import scipy.io
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from kNNClass import kNN
from MVGClass import MVG


def main():
    segments = np.arange(0, 10)

    split = np.arange(7000, 10000, 800)

    accuracies_knn = np.zeros(len(segments))
    accuracies_mvg = np.zeros(len(segments))

    time_knn = np.zeros(len(split))
    time_mvg = np.zeros(len(split))

    mat_file = sys.path[0] + "/hw1data.mat"
    mat_data = scipy.io.loadmat(mat_file)

    X = mat_data['X']
    Y = mat_data['Y']

    '''
    x = 0
    for s in segments:
        print(x)
        ind = np.random.permutation(len(X))
        X = X[ind]
        Y = Y[ind]
        X_training_data = X[0:9000, :]

        Y_training_data = Y[0:9000, 0]

        MVG_classifier = MVG(X_training_data, Y_training_data)

        count = 0
        correct = 0

        for test_var in range(9000,10000):
            count += 1
            test = MVG_classifier.classify(X[test_var])
            if test == Y[test_var][0]:
                correct += 1
        accuracies_mvg[x] = correct / count
        x += 1


    plt.plot(segments, accuracies_mvg, 'o', label='Multivariate Gaussian Classifier')

    X = mat_data['X']
    Y = mat_data['Y']

    x = 0
    for s in segments:
        print(x)
        ind = np.random.permutation(len(X))
        X = X[ind]
        Y = Y[ind]
        X_training_data = X[0:9000, :]

        Y_training_data = Y[0:9000, 0]

        kNN_classifier = kNN(X_training_data, Y_training_data, metric='L2')

        count = 0
        correct = 0

        for test_var in range(9000,10000):
            count += 1
            output =  kNN_classifier.classify(X[test_var])
            if output == Y[test_var][0]:
                correct += 1
        accuracies_knn[x] = correct / count
        x+=1

    plt.plot(segments, accuracies_knn, 'x', label='kNN Clasifier')

    plt.show()
    '''
    '''
    # size of training data vs accuracy
    x = 0
    for s in split:
        print(s)
        X_training_data = X[0:s, :]
        Y_training_data = Y[0:s, 0]

        t0 = time.time()
        kNN_classifier = kNN(X_training_data, Y_training_data, metric='L2')

        count = 0
        correct = 0
        for test_var in range(s, 10000):
            count += 1
            output = kNN_classifier.classify(X[test_var])
            if output == Y[test_var][0]:
                correct += 1
        x += 1
        t1 = time.time()
        time_knn[x] = t1 - t0
    plt.plot(split, time, 'x', label='kNN Classifier')
    '''
    '''
    # Simple example to classify given a classifier clf
    t0 = time.time()
    X_training_data = X[0:8000, :]
    Y_training_data = Y[0:8000, 0]
    # clf = MVG(train_X, train_Y, max_features=250)
    kNN_classifier = kNN(X_training_data, Y_training_data, metric='L2')
    test = kNN_classifier.classify(X[9001])

    t1 = time.time()
    print("We got: {}, the correct answer was {}. It took {} seconds".format(test, Y[9001][0], t1 - t0))
    '''

    # size of training data vs accuracy
    '''
    x = 0
    for s in split:
        print(s)
        X_training_data = X[0:s, :]
        Y_training_data = Y[0:s, 0]

        t0 = time.time()
        MVG_classifier = MVG(X_training_data, Y_training_data)

        count = 0
        correct = 0
        for test_var in range(s, 10000):
            count += 1
            test = MVG_classifier.classify(X[test_var])
            if test == Y[test_var][0]:
                correct += 1

        x += 1
        t1 = time.time()
        time[x] = t1 - t0

    plt.plot(split, accuracies, 'o', label='Multivariate Gaussian')
    plt.show()
    '''

    t0 = time.time()
    # shuffles the data

    x_axis = []
    accuracies = []
    l1 = []
    l2 = []
    l3 = []
    for i in range(1, 7):
        length = 5000 + 500 * i
        x_axis.append(length)

        ind = np.random.permutation(len(X))
        X = X[ind]
        Y = Y[ind]
        X_training_data = X[0:length, :]
        Y_training_data = Y[0:length, 0]

        kNN_L1 = kNN(X_training_data, Y_training_data, metric='L1')
        kNN_L2 = kNN(X_training_data, Y_training_data, metric='L2')
        kNN_Linf = kNN(X_training_data, Y_training_data, metric='L_inf')

        count = 0
        correct_L1 = 0
        correct_L2 = 0
        correct_Linf = 0
        for test_var in range(8000, 8050):
            count += 1
            test_1 = kNN_L1.classify(X[test_var])
            test_2 = kNN_L2.classify(X[test_var])
            test_inf = kNN_Linf.classify(X[test_var])

            if test_1 == Y[test_var][0]:
                correct_L1 += 1
            if test_2 == Y[test_var][0]:
                correct_L2 += 1
            if test_inf == Y[test_var][0]:
                correct_Linf += 1

        l1.append(correct_L1 / count)
        l2.append(correct_L2 / count)
        l3.append(correct_Linf / count)

        print("L1 : {}, L2: {}, L3: {}".format(correct_L1 / count, correct_L2 / count, correct_Linf / count))

    plt.plot(x_axis, l1, 'g^', label='L1')
    plt.plot(x_axis, l2, 'bs', label='L2')
    plt.plot(x_axis, l3, 'r--', label='L_inf')
    plt.show()

    t1 = time.time()
    print(t1 - t0)


if __name__ == '__main__':
    main()
