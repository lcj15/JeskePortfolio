# TODO create cluster (validity) metrics


# Todo add comments
# Todo make sure "bad swaps" are not accepted in KMediods
# Todo refactor redundant code into util
# Todo add notebook(s)

def congressional_distance(a, b):
    """
    Function that calculates dissimilarity between congressmen according to their
    voting record on 16 issues in 1984
    :param a: voting record of congressman a
    :param b: voting record of congressman b
    :return: dissimilarity between the two congressmen
    """
    # gather the indices of where both members a and b voted on the same issues
    ind = np.intersect1d(np.nonzero(a), np.nonzero(b))

    # if member a never voted on the same issue as member b, the dissimilarity is 0.5
    if len(ind) == 0:
        return 0.5
    # dissimilarity is the number of times a and b voted differently on the same issue
    # divided by the number of times they voted on the same issue
    return np.sum(np.not_equal(a[ind], b[ind])) / len(ind)


def random_mediod_initialization(D, n_init, k):
    """
    Randomly initialize the k-mediods 'n_init' times
    :param D: data
    :param n_init: number of times to repeat the initialization
    :param k: number of clusters
    :return: initial set of mediods
    """
    min_indices = random.sample(range(len(D)), k)
    D_m = D[np.ix_(min_indices)].T
    min_tightness = np.sum(np.min(D_m, axis=1))
    for i in range(n_init):
        temp_indices = random.sample(range(len(D)), k)
        temp_D = D[np.ix_(temp_indices)].T
        temp_tightness = np.sum(np.min(temp_D, axis=1))
        if temp_tightness < min_tightness:
            min_tightness = temp_tightness

            min_indices = temp_indices
    return min_indices


def confusion_matrix(pred_labels, actual_labels):
    """
    Generate a confusion matrix to compare clustering results to the class labels of
    the data
    :param pred_labels: Labels predicted by model
    :param actual_labels: True labels of the data
    :return: Confusion matrix showing the relationship between cluster membership and class membership
    """
    assert (len(np.unique(pred_labels)) == len(np.unique(actual_labels)))
    kk = len(np.unique(pred_labels))
    C = np.zeros((kk, kk))
    for i in range(kk):
        for j in range(kk):
            C[i][j] = len(np.intersect1d(*np.where(pred_labels == i), np.where(actual_labels == j)))
    return C


def random_init(data, k):
    """
    Takes a dataset data, randomly assigns a cluster index 0 through k for each
    point and returns the average of each of those clusters.
    :param data: data
    :param k: number of clusters
    :return:
    """
    return [np.mean(data[np.ix_(j)], axis=0) for j in
            [np.where(rd.randint(low=0, high=k, size=len(data)) == i)[0] for i in range(k)]]


def seed_init(data, k):
    """
    Randomly picks k data points without replacement
    :param data: data
    :param k: number of clusters
    :return: three randomly sampled data points
    """
    return [data[i] for i in rd.randint(0, len(data), k)]


# todo refactor this, very redundant
def initialize(algo, data, k):
    """
    returns k cluster centroids returned by the specified algorithm
    """
    if algo == "seed":
        return seed_init(data, k)
    elif algo == "random":
        return random_init(data, k)
    else:
        raise ValueError("Invalid name for algorithm")


def repeat_init(data, k, n, init):
    """
    Initializes k centroids n times and keeps the "tightest" initialization
    :param data:
    :param k:
    :param n:
    :param init:
    :return:
    """
    # initially, the minimum is the first
    min_clusters = initialize(init, data, k)
    min_tightness = np.sum(distance_matrix(min_clusters, data) ** 2)
    # repeat n times
    for i in range(n):
        a = initialize(init, data, k)
        tightness = np.sum(distance_matrix(min_clusters, data) ** 2)
        if tightness < min_tightness:
            # swap if we found tighter indices
            min_tightness = tightness
            min_clusters = a
    return min_clusters


def overall_tightness(Dm):
    """
    Calculates the tightness from a rectangular distance matrix between cluster
    centers and the dataset
    :param Dm:
    :return:
    """
    return np.sum(np.min(Dm, axis=1) ** 2)


class KMediods:
    """
    Object-Oriented Programming Approach to K-Mediods Algorithm
    """
    def __init__(self, n_clusters=3, initialization="random", n_init=20, max_iter=300, tau=0):
        self.tightness = None
        self.I_assign = None
        self.cluster_centers = None
        self.mediod_indices = None
        self.n_clusters = n_clusters

        self.initialization = initialization
        self.n_init = n_init
        self.max_iter = max_iter
        self.tau = tau
        self.n_iter_ = 0

    def fit(self, X, funct=distance):
        """
        sets the clusters and cluster indices for each data point in X with a predefined
        distance function
        :param X: dataset to fit the model to
        :param funct: specification of distance metric
        :return: None
        """
        # initialize the distance function from every point to every point once
        self.D = distance_matrix(X, X, funct)

        # k is the number of clusters
        k = self.n_clusters
        # initialize by randomly sampling k data points
        if self.initialization == "random":
            mediod_indices = random_mediod_initialization(self.D, self.n_init, k)
        else:
            mediod_indices = seed_init(self.D, k)

        # TODO include cases for other forms of initialization

        tightness = []

        # distances to current mediods
        D_m = self.D[np.ix_(mediod_indices)].T
        # assignment to each cluster
        I_assign = np.argmin(D_m, axis=1)
        # the ith index of "clusters" denotes the indices for data points in cluster i
        clusters = [np.where(I_assign == i)[0] for i in range(k)]

        # overall tightness
        Q = overall_tightness(D_m)
        tightness.append(Q)
        # collapse the square distance matrix to a vector by summing its rows
        sums = [np.sum(self.D[np.ix_(clusters[i])], axis=0) for i in range(k)]
        # set change to infinity

        change = np.inf
        # run until the tightness does not change or max iterations is exceeded
        while change > self.tau and self.n_iter_ < self.max_iter:
            # store the 'previous' tightness
            tightness1 = Q
            # take the minimum "collapsed sum" for each data point
            mediod_indices = [np.argmin(sums[i]) for i in range(k)]
            # index slicing of precalculated distance matrix
            D_m = self.D[np.ix_(mediod_indices)].T
            # assignment to each cluster
            I_assign = np.argmin(D_m, axis=1)

            # calculate the tightness
            Q = overall_tightness(D_m)
            tightness.append(Q)
            tightness2 = Q
            # create a list of lists where the ith inner list stores the data indices for the
            # ith cluster
            clusters = [np.where(I_assign == i)[0] for i in range(k)]
            # sum each local distance matrix
            sums = [np.sum(self.D[np.ix_(clusters[i])], axis=0) for i in range(k)]
            self.n_iter_ += 1
            change = (abs(tightness2 - tightness1) / tightness1)

        self.mediod_indices = mediod_indices
        self.cluster_centers = clusters
        self.I_assign = I_assign
        self.tightness = tightness
