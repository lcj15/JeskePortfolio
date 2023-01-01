from Supervised.Classifier import Classifier as Classifier
import numpy as np
from util import pdistance_matrix
from util import custom_distance_matrix


class KNearestNeighbor(Classifier):
    def __init__(self, k):
        """
        Initializes the KNN, fixes the value of k.  KNN is a lazy classifier.  Fitting is merely the recording of data
        and labels seen during fitting.
        :param k: number of neighbors used for predictions, should be odd
        """
        self.trainy = None
        self.trainX = None
        self.k = k

    def fit(self, X, y):
        """
        Fits the model to the training data
        :param X: training data
        :param y: training labels
        :return: None
        """
        self.trainX = X
        self.trainy = y

    def predict(self, X):
        """
        Make predictions to the testing data according to the KNN algorithm'
        :param X: data
        :return: predictions of the data based on the model
        """
        # create a distance matrix between the testing data and the training data
        # sort that distance matrix such that for each of the testing data points,
        # each training data point is sorted in order from closest to furthest
        # we want the index of the data that is closest to it, not the data itself
        # only keep the 'k' nearest indices
        dm = np.argsort(custom_distance_matrix(X, self.trainX))[:, :self.k]

        # retrieve the class label for each of the k nearest indices for each test
        # data point
        votes = np.array([self.trainy[np.ix_(dm[i])] for i in range(len(X))])

        # for each testing data point, return the class that the majority of its neighbors
        # belong to
        return np.array([np.argmax(np.bincount(votes[i])) for i in range(len(X))])
