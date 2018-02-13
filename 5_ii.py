import scipy
import scipy.io
import sys
import numpy as np
import math
import time


def main():
    mat_file = read_mat_file()

    x = mat_file['X']
    y = mat_file['Y']

    t0 = time.time()
    short_list = find_k_NN(10, x[5252], x)
    for i in short_list:
        print(y[i[1]])
    t1 = time.time()

    print("Time it took: {}".format(t1 - t0))


def find_k_NN(k, data_point, x):
    # Store the sorted list of indices that map to the k shortest known Euclidean distances.
    # Shortest will contian k tuples of (distance, index)
    shortest = [(float("inf"), 0)] * k
    print(len(x))
    for pic_index in range(len(x)):
        ed = calc_euclidean_distance(data_point, x[pic_index])
        if ed <= shortest[k - 1][0]:
            # we have a new shortest distance
            shortest.pop()
            found = False
            for i in range(len(shortest)):
                if shortest[i][0] >= ed:
                    shortest.insert(i, (ed, pic_index))
                    found = True
                    break
            if not found:
                shortest.append((ed, pic_index))
    print(shortest)
    return shortest


def calc_euclidean_distance(dp_1, dp_2):
    distance_sum = 0

    for i in range(len(dp_1)):
        distance_sum += (float(dp_1[i]) - float(dp_2[i]))**2

    distance = math.sqrt(distance_sum)
    if distance < 2:
        print("Equal")

    print(distance)
    return distance


def read_mat_file():
    mat_file = sys.path[0] + "/hw1data.mat"

    mat = scipy.io.loadmat(mat_file)

    return mat


if __name__ == '__main__':
    main()
