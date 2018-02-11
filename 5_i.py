import scipy
import scipy.io
import sys
import numpy as np


def main():
    mat_file = read_mat_file()

    x = mat_file['X']
    y = mat_file['Y']

    means = get_means(x, y)
    print("Means:\nType: {} \nShape: {}".format(type(means), means.shape))
    stds = get_stds(means, x, y)


def get_means(x, y):
    sums_arr = np.zeros(shape=(10, len(x[0])), dtype=float)
    temp = sums_arr[0].reshape(28, 28)
    print(sums_arr[0].shape)

    # Summation
    for pic_index in range(len(x)):
        num = y[pic_index]
        print(sums_arr[num].shape)
        print(type(sums_arr[num]))
        sums_arr[num] += x[num]

    # Divide by n
    sums_arr /= len(x)

    return sums_arr


def get_stds(means, x, y):
    stds_arr = [np.asmatrix(np.zeros(shape=(28, 28), dtype=float))] * 10
    print(type(stds_arr[0]))
    for pic_index in range(len(x)):
        num = y[pic_index]
        num_index = num[0]
        reshaped_x = x[num].reshape(28, 28)
        reshaped_mean = means[num].reshape(28, 28)

        a = (reshaped_x - reshaped_mean)
        b = np.transpose(reshaped_x - reshaped_mean)
        c = np.asmatrix(a * b)
        print(num_index)
        print("stds_arr data:\tShape: {}\tType: {}".format(stds_arr[num_index].shape, type(stds_arr[num_index])))
        print("C data:\tShape: {}\tType: {}".format(c.shape, type(c)))
        stds_arr[num_index] = np.add(stds_arr[num_index], c)

    # divide by n
    for i in range(len(stds_arr)):
        stds_arr[i] /= len(x)

    return stds_arr


def read_mat_file():
    mat_file = sys.path[0] + "/hw1data.mat"

    mat = scipy.io.loadmat(mat_file)

    return mat


if __name__ == '__main__':
    main()
