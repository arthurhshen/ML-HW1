import numpy as np
from scipy.stats import multivariate_normal


class MVG:
    def __init__(self, x, y):
        self.X = x
        self.Y = y
        self.model = {}
        self.preprocess(self.X, self.Y)

    def classify(self, point):
        max_prob = float('-inf')
        for num in self.model.keys():
            mean, std, rv, prior, indices = self.model[num]

            # Only select the max_indices for this number
            point_filtered = point[indices]
            z_score = (point_filtered - mean) / std

            prob = rv.pdf(z_score) * prior
            if prob > max_prob:
                max_prob = prob
                label = num

        return label

    def preprocess(self, X, Y):
        # We will select the 200 "best" features for each number
        features = -200
        labels = np.unique(Y)

        for num in labels:
            X_num = X[Y == num]

            # Select the best 200 features
            complete_cov = np.cov(X_num, rowvar=False)
            # For each column, take the average
            avg_cov = np.mean(complete_cov, axis=1)

            max_indices = np.argpartition(avg_cov, features)[features:]
            X_processed = X_num[:, max_indices]

            # Normalize processed matrix
            # https://stackoverflow.com/questions/31152967/normalise-2d-numpy-array-zero-mean-unit-variance
            # had to reshape means - was (250, 0)
            mean = np.mean(X_processed, axis=0)
            std = X_processed.std(axis=0)
            X_normed = (X_processed - mean) / std

            # add λI to the covariance matrix for some small λ - set λ to 0.5
            means = np.mean(X_normed, axis=0)
            normed_cov = np.cov(X_normed, rowvar=False)
            normed_cov = normed_cov + 0.5 * np.identity(200)

            # Calculate the distribution for this number
            rv = multivariate_normal(mean=means, cov=normed_cov)

            prior = X_num.shape[0] / X.shape[0]

            self.model[num] = (mean, std, rv, prior, max_indices)
