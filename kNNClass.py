import scipy.spatial.distance as sp_dist
import numpy as np
from scipy.stats import mode


class kNN:
    def __init__(self, x, y, k=10, metric=None):
        self.X = x
        self.Y = y
        self.metric = sp_dist.euclidean
        self.k = k
        print(k)

    def train(self, x=None, y=None, k=None, metic=None):
        if x is not None:
            self.X = x
        if y is not None:
            self.Y = y
        if k is not None:
            self.k = k

        if metric == "L1":
            pass
        if metric == "L2":
            self.metric = sp_dist.euclidean
        if metric == "L_inf":
            pass

    # calculates the distance between the 2 points
    def get_dist_wrapper(self, p1, p2):
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.astype.html
        # Seems to be an issue with the unsigned int type - casting to int
        return self.metric(p1.astype(int), p2.astype(int))

    def calc_distance(self, point):
        dists = np.apply_along_axis(self.get_dist_wrapper, 1, self.X, point)
        # Sources: https://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.argpartition.html
        y_indices = np.argpartition(dists, self.k, axis=0)
        y_values = self.Y[y_indices[0:self.k]]
        return mode(y_values)[0][0]

    def classify(self, point):
        return self.calc_distance(point)
