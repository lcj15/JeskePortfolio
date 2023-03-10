import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import util

from Classifier import Classifier as Classifier
from Supervised.DistanceClassifier import DistanceClassifier as DC


# TODO add predict method using either KNN or distance classifier
# TODO add comments

class LDA(Classifier):
    """
    Linear Discriminant Analysis based on eigen decomposition.
    """

    def __init__(self, n, clf: Classifier = DC()):
        """
        Initialize the model.
        :param n: number of dimensions to transform to
        """
        self.Q = None
        self.k = None
        self.n = n
        self.clf = clf
        assert (n >= 1)

    def fit(self, X, y):
        """
        Fit the model to the data
        :param X: training data
        :param y: training labels
        :return: none
        """

        # number of distinct classes
        k = len(np.unique(y))
        assert (self.n <= k - 1)

        # partition data based on labels
        cluster_indices = [np.where(y == i)[0] for i in range(k)]
        cluster_data = [X[i] for i in cluster_indices]
        cluster_means = [np.mean(i, axis=0) for i in cluster_data]
        cluster_cardinality = [len(i) for i in cluster_indices]
        locally_centered_data = [cluster_data[i] - cluster_means[i] for i in
                                 range(k)]

        # within_cluster_centered_data
        x_w = np.matrix([val for sublist in locally_centered_data for val in
                         sublist])
        s_w = x_w.T @ x_w
        x_bar_l = [np.tile(cluster_means[i], (cluster_cardinality[i], 1)) for i in range(k)]
        x_bar = util.vector_stack(x_bar_l)
        x_bar_c = x_bar - np.mean(X, axis=0)
        s_b = x_bar_c.T @ x_bar_c

        eigs_sw = la.eigvals(s_w)

        # if matrix is not positive definite regularize it
        if np.min(eigs_sw) < 0:
            largest_eig = np.max(eigs_sw)

            epsilon = 1e-8 * largest_eig
            s_w = s_w + epsilon * np.identity(len(s_w))

        K = la.cholesky(s_w)
        A = K.I @ s_b @ K.I.T
        W = np.matrix(util.sorted_eig(A)[:, :self.n])
        Q = np.array(K.I.T @ W).T
        self.k = k
        self.Q = Q

        if self.clf:
            self.clf.fit(self.transform(X.T).T, y)

    def transform(self, X):
        """

        :param X:
        :return:
        """
        return self.Q @ X

    def plot(self, X, y):
        """
        Produce LDA plot
        :param X:
        :param y:
        :return:
        """
        Z = self.fit_transform(X, y)
        plt.scatter(*Z.T[:2], c=y)
        plt.title("LDA Reduced Data")
        plt.show()  # is this line necessary?

    def fit_transform(self, X, y):
        """
        Fits model to the data and then transforms it
        :param X:
        :param y:
        :return:
        """
        self.fit(X, y)
        return self.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        :param X:
        :return:
        """
        return self.clf.predict(self.transform(X.T).T)
