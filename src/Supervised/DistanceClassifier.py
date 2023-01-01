import numpy as np
import scipy.spatial.distance as distance

import util
from Supervised.Classifier import Classifier as Classifier


class DistanceClassifier(Classifier):
    """
    Implementation of the Distance Classifier
    """

    def __init__(self, func=distance.euclidean):
        """
        initialize the algorithm by setting the distance metric to be used
        :param func: method of comparison
        """

        self.cluster_means = None
        self.func = func

    def fit(self, X, y):
        """
        fit the model to the training data
        :param X: training data
        :param y: training labels
        :return: None
        """
        # partition the data by class, then calculate the centroid of each class
        k = len(np.unique(y))
        cluster_indices = [np.where(y == i)[0] for i in range(k)]
        cluster_data = [X[i] for i in cluster_indices]
        cluster_means = [np.mean(i, axis=0) for i in cluster_data]
        self.cluster_means = cluster_means

    def predict(self, X):
        """
        Make predictions on the testing data based on which centroid it is closer too
        :param X: test data
        :return: assigns the class label based on the closest class cluster mean
        """
        # create a distance matrix between the test data and the centroids of the training data
        dm = util.custom_distance_matrix(X, self.cluster_means, self.func)
        # return the class centroid that each data point is closer too
        return np.argmin(dm, axis=1)
