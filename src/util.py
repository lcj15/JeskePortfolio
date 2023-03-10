import scipy.spatial.distance
from scipy.spatial.distance import pdist, squareform, cdist, euclidean
from scipy.spatial import minkowski_distance, distance_matrix
import numpy as np
import numpy.linalg as la

# todo this does not work, change it



def dm(x,y,func=euclidean):
    return cdist(x,y,func)


def pdistance_matrix(data, metric="euclidean", **kwargs):
    """
    Efficiently computes pairwise distance matrix by utilizing symmetric property and the
    pdist function from scipy
    :param metric:
    :param data: data vectors from a dataset
    :return: pairwise distance matrix
    """
    l = len(data)
    a = np.zeros((l, l))
    a[np.tril_indices(l, -1)] = pdist(data, metric=metric)
    return a + np.tril(a, -1).T


# todo change name to non-symmetric
def custom_distance_matrix(x, y, func=scipy.spatial.distance.euclidean):
    """
    Creates a distance matrix between two sets of data according to a specified
    distance metric.  Intended for custom similarity metrics.
    :param x: first group of data
    :param y: second group of data
    :param func: similarity function
    :return: pairwise distance matrix between every point in x and every point in y
    """

    # Check to see if the distance matrix will be square
    is_symmetric = np.alltrue(x == y)

    # initialize return matrix
    m = np.zeros((len(x), len(y)))

    if is_symmetric:
        # Only compute lower triangular entries (diagonal entries are 0, matrix is symmetric)
        for i1, j1 in enumerate(x):
            for i2, j2 in enumerate(y):
                if i1 < i2:
                    m[i1][i2] = func(j1, j2)
        return m + np.tril(m, -1).T

    else:
        # Compute all entries (useful for clustering distance matrices)
        for i1, j1 in enumerate(x):
            for i2, j2 in enumerate(y):
                m[i1][i2] = func(j1, j2)
        return m


def vector_stack(lst):
    """
    Flattens a list of lists
    :param lst: list of lists where each of the outer lists corresponds to class label
    :return:
    """
    return np.matrix([val for sublist in lst for val in sublist])


def sorted_eig(mat):
    """
    Returns eigenvectors in descending value with respect to the eigenvalues
    :param mat: data matrix
    :return: matrix of stacked eigenvectors in decreasing order of eigenvalues
    """
    eigenValues, eigenVectors = la.eigh(mat)
    idx = eigenValues.argsort()[::-1]
    eigenVectors = eigenVectors[:, idx]
    return eigenVectors


def accuracy(pred_labels, actual_labels):
    """
    Returns accuracy of predictions of a classifier
    :param pred_labels: labels predicted by a classifier
    :param actual_labels: true labels of data
    :return: accuracy of classifier
    """
    assert (len(pred_labels) == len(actual_labels))
    return np.sum(pred_labels == actual_labels) / len(actual_labels)
