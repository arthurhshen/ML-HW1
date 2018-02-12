import scipy
import scipy.io
import sys
import numpy as np
import math




def main():
    mat_file = read_mat_file()

    x_train = mat_file['X'][0:9000]
    y_train = mat_file['Y'][0:9000]

    x_test = mat_file['X'][9000:]
    y_test = mat_file['Y'][9000:]
    
    means = get_means(x_train, y_train)
    stds = get_covs(means, x_train, y_train)
    priors = get_class_priors(y_train)
    
    probabilities = []

    for x in range(len(means)):
        a = float(np.linalg.det(stds[x]))
        print(a)
        b = 1.0 / math.sqrt(a)

        diff = x_test[0] - means[x]
        diff_T = np.transpose(diff)
        std_inv = np.linalg.inv(stds[x])

        exp = math.exp(-0.5*(diff_T.dot(std_inv)).dot(diff))
        prob = b*exp
        probabilities.append(prob)
    print(probabilities)
    


def get_class_priors(y_train):
    occurrences = {}
    for label in y_train:
        if label[0] in occurrences:
            occurrences[label[0]] += 1
        else:
            occurrences[label[0]] = 1

    priors = {}

    for label in occurrences:
        priors[label] = occurrences[label] / float(len(y_train))

    return priors


def get_means(x_train, y_train):
    counts = np.zeros(10)
    sums_arr = [np.asmatrix(np.zeros(shape=(784,), dtype=float))] * 10

    # Summation
    for i in range(len(x_train)):
        label = y_train[i][0]
        sums_arr[label] += x_train[i]
        counts[label] += 1

    # Divide by n
    for j in range(len(sums_arr)): 
        sums_arr[j] = sums_arr[j] / counts[j]

    return sums_arr


def get_covs(means, x_train, y_train):
    covs_arr = np.zeros((10,784,784), dtype=float)
    counts = np.zeros(10)
    for i in range(len(x_train)):
        label = y_train[i][0]

        a = x_train[i] - means[label]
        b = np.transpose(a)
        c = a.dot(b)
        covs_arr[label] += c
        counts[label] += 1

    # divide by n
    for j in range(10): 
        covs_arr[j] = covs_arr[j] / counts[j]

    return covs_arr


def read_mat_file():
    mat_file = sys.path[0] + "/hw1data.mat"

    mat = scipy.io.loadmat(mat_file)


    return mat


if __name__ == '__main__':
    main()
