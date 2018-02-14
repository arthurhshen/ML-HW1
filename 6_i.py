import scipy
import scipy.io
import sys
import numpy as np
import math

class Node:
    def __init__(self):
        self.left = null
        self.right = null
        self.label = null
        self.threshold = 0


def main():
    mat_file = read_mat_file()

    x_train = mat_file['X'][0:9000]
    y_train = mat_file['Y'][0:9000]

    x_test = mat_file['X'][9000:]
    y_test = mat_file['Y'][9000:]
    



def id3(input_labels, output_labels, x_train, y_train):

    current = Node()
    for x in range(784):
        for t in range(255):






    current.label = max(get_class_priors())d




def get_class_priors(y_train):
    occurrences = np.zeros(10)
    for label in y_train:
        occurrences[label] += 1

    priors = np.zeros(10)

    for x in range(10):
        priors[x] = occurrences[x] / float(len(y_train)

    return priors


def read_mat_file():
    mat_file = sys.path[0] + "/hw1data.mat"

    mat = scipy.io.loadmat(mat_file)


    return mat


if __name__ == '__main__':
    main()

