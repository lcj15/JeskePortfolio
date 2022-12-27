import scipy.spatial.distance


class KMeans:
    """
    Object-Oriented Programming Approach to K-Means Algorithm based on Lloyd's algorithm
    """
    def __init__(self, n_clusters, initialization="random", n_init=20, max_iter=300, tau=0):
        self.I_assign = None
        self.tightness = None
        self.cluster_centers = None
        self.n_clusters = n_clusters
        self.init = initialization
        self.n_init = n_init
        self.max_iter = max_iter
        self.tau = tau
        self.n_iter_ = 0

    def fit(self, X, funct=scipy.spatial.distance.euclidean):
        """
        Trains the model to the dataset according to the distance function
        :param X:
        :param funct: how distance is calculate
        :return:
        """
        k = self.n_clusters
        # initialize the centroids
        cluster_centers = repeat_init(X, k,self.n_init, self.init)
        # bookkeeping
        tightness = []
        # calculate distance from cluster centers to all the data
        dm = distance_matrix(X, cluster_centers,funct)
        # assign cluster indices to each data point based on cluster proximity
        I_assign = np.argmin(dm, axis = 1)
        # calculate tightness
        Q = np.sum(np.min(dm, axis=1)**2)
        tightness.append(Q)
        # set temp to infinity
        change = np.inf
        # stops when the change is relatively small or when max iterations is exceeded
        while change > self.tau and self.n_iter_ < self.max_iter:
            # store to calculate percent change
            tightness1 = Q
            # local distance matrices
            D_ls = [np.where(I_assign == i) for i in range(k)]
            # calculate cluster means based on local matrices
            cluster_centers = [np.average(X[D_ls[i]], axis=0) for i in range(k)]



            # recalibrate the distance matrix
            dm = distance_matrix(X, cluster_centers,funct)

            # recalculate the tightness
            Q = np.sum(np.min(dm, axis=1)**2)
            tightness2 = Q
            # add it to the records
            tightness.append(np.sum(np.min(dm, axis=1)**2))
            # reassign cluster indices
            I_assign = np.argmin(dm, axis = 1)
            # calculate the change in tightness
            change = (abs(tightness2 - tightness1) / tightness1)

            # increment counter
            self.n_iter_ += 1
            self.cluster_centers = cluster_centers
            self.tightness = tightness
            self.I_assign = I_assign