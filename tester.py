import sys
import scipy.io
import numpy as np
import random
import time
import matplotlib.pyplot as plt
from kNNClass import kNN
from MVGClass import MVG


def main():

    split = np.arange(5000,9001,500)
    segments = np.arange(0,10)
    print(segments)

    accuracies = np.zeros(len(split))
    mat_file = sys.path[0] + "/hw1data.mat"
    mat_data = scipy.io.loadmat(mat_file)
    N = len(mat_data['X'])

    X = mat_data['X']
    Y = mat_data['Y']

    '''
    for s in segment:
        X_training_data = X[s:s+2000, :]
        Y_training_data = X[s:s+2000, 0]

        kNN_classifier = kNN(X_training_data, Y_training_data, metric='L2')

        count = 0
        correct = 0

        for x in range(0,s):
            count += 1
            output =  kNN_classifier.classify(X[test_var])
            if output == Y[test_var][0]:
                correct += 1

        for x in range(s+2000,10000):
            count += 1
            output =  kNN_classifier.classify(X[test_var])
            if output == Y[test_var][0]:
                correct += 1

    plt.plot(segments, accuracies, 'o') 
    plt.show()

    x = 0
    print(Y)
    for s in segments:
        shuffling = np.arange(10000)
        np.random.shuffle(shuffling)

        X = X[shuffling]
        Y = Y[shuffling]
        X_training_data = X[0:8000, :]
        Y_training_data = X[0:8000, 0]

        MVG_classifier = MVG(X_training_data, Y_training_data)

        count = 0
        correct = 0

        for test_var in range(8000,10000):
            count += 1
            test = MVG_classifier.classify(X[test_var])
            if test == Y[test_var][0]:
                correct += 1
        accuracies[x] = correct / count
        print(count, correct)
        x += 1


    plt.plot(segments, accuracies, 'o') 
    plt.show()
    '''
    '''
    # size of training data vs accuracy 
    x = 0
    for s in split:
        print(s)
        X_training_data = X[0:s, :]
        Y_training_data = Y[0:s, 0]

    #t0 = time.time()
        kNN_classifier = kNN(X_training_data, Y_training_data, metric='L2')

        count = 0
        correct = 0
        for test_var in range(s, 10000):
            count += 1
            output =  kNN_classifier.classify(X[test_var])
            if output == Y[test_var][0]:
                correct += 1
        accuracies[x] = correct / count
        x += 1
    #t1 = time.time()
    plt.plot(split, accuracies, 'o') 
    plt.show()
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
    
    x = 0
    for s in split:
        print(s)
        X_training_data = X[0:8000, :]
        Y_training_data = Y[0:8000, 0]

    #t0 = time.time()
        MVG_classifier = MVG(X_training_data, Y_training_data)

        count = 0
        correct = 0
        for test_var in range(8000, 10000):
            count += 1
            test = MVG_classifier.classify(X[test_var])
            if test == Y[test_var][0]:
                correct += 1
        print(correct, count)
        accuracies[x] = correct / count
        x += 1
    #t1 = time.time()
    plt.plot(split, accuracies, 'o') 
    plt.show()
    

if __name__ == '__main__':
    main()
