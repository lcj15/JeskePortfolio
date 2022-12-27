
# TODO create cluster (validity) metrics

def distance(a,b):
    '''
    returns the euclidean norm between two vectors
    '''
    return np.linalg.norm(a-b)
def distance_matrix(data1, data2, funct=distance):
    '''
    creates a distance matrix between two sets of data according to a specified distance function
    '''
    m = np.zeros((len(data1),len(data2)))
    for i1, j1 in enumerate(data1):
        for i2, j2 in enumerate(data2):
            m[i1][i2] = funct(j1,j2)
    return m

# Todo add comments
# Todo make sure "bad swaps" are not accepted in KMediods
# Todo refactor redundant code into util
# Todo add notebook(s)

def congressional_distance(a,b):
    '''
    function that calculates dissimilarity between congressmen according to their
    voting record on 16 issues in 1984
    '''
    # gather the indices of where both members a and b voted on the same issues
    ind = np.intersect1d(np.nonzero(a), np.nonzero(b))

    # if member a never voted on the same issue as member b, the dissimilarity is
    0.5
    if(len(ind) == 0):
        return 0.5
    # dissimilarity is the number of times a and b voted differently on the same
    issue
    # divided by the number of times they voted on the same issue
    return sum( np.not_equal(a[ind], b[ind]) ) / len(ind)

def random_mediod_initalization(D, n_init, k):
    '''
    randomly initialize the k mediods 'n_init' times
    '''
    min_indices = random.sample(range(len(D)),k)
    D_m = D[np.ix_(min_indices)].T
    min_tightness = np.sum( np.min(D_m, axis=1))
    for i in range(n_init):
        temp_indices = random.sample(range(len(D)),k)
        temp_D = D[np.ix_(temp_indices)].T
        temp_tightness = np.sum( np.min(temp_D, axis=1))
        if temp_tightness < min_tightness:
            min_tightness = temp_tightness

            min_indices = temp_indices
    return min_indices

def confusion_matrix(pred_labels, actual_labels):
    '''
    generate a confusion matrix to compare clustering results to the class labels of
    the data
    '''
    assert(len(np.unique(pred_labels)) == len(np.unique(actual_labels)))
    kk = len(np.unique(pred_labels))
    C = np.zeros((kk,kk))
    for i in range(kk):
        for j in range(kk):
            C[i][j] = len(np.intersect1d(*np.where(pred_labels == i), np.where(actual_labels==j)))
    return C

def random_init(data, k):
    '''
    Takes a dataset data, randomly assigns a cluster index 0 through k for each
    point and
    returns the average of each of those clusters
    '''
    return [np.mean(data[np.ix_(j)], axis=0)  for j in [np.where( \
        rd.randint(low=0,high=k,size=len(data)) == i)[0] for i in range(k)]]

def seed_init(data, k):
    '''
    Randomly picks k data points without replacement
    '''
    return [data[i] for i in rd.randint(0,len(data),k)]

def initialize(algo, data, k):
    '''
    returns k cluster centroids returned by the specified algorithm
    '''
    if(algo == "seed"):
        return seed_init(data,k)
    elif(algo == "random"):
        return random_init(data,k)
    else:
        raise ValueError("Invalid name for algorithm")

def repeat_init(data, k, n, init):
    '''
    initializes k centroids n times and keeps the "tightest" initialization
    '''
    # initially, the minimum is the first
    min_clusters = initialize(init, data, k)
    min_tightness = np.sum( distance_matrix(min_clusters, data)**2 )
    # repeat n times
    for i in range(n):
        a = initialize(init, data,k)
        tightness = np.sum( distance_matrix(min_clusters, data)**2 )
        if(tightness < min_tightness):
            # swap if we found tighter indices
            min_tightness = tightness
            min_clusters = a
    return min_clusters

def overall_tightness(Dm):
    '''
    Calculates the tightness from a rectangular distance matrix between cluster
    centers
    and the dataset
    '''
    return np.sum( np.min(Dm, axis=1)**2 )

class KMediods:
    ''' Object Oriented Programming Approach to K-Mediods Algorithm'''
    def __init__(self, n_clusters=3, initialization="random", n_init=20, max_iter=300, tau=0):
        self.n_clusters = n_clusters

        self.initialization = initialization
        self.n_init = n_init
        self.max_iter = max_iter
        self.tau = tau
        self.n_iter_ = 0

    '''
    sets the clusters and cluster indices for each data point in X with a predefined
    distance
    function
    '''
    def fit(self, X, funct = distance):
        # initialize the distance function from every point to every point once
        self.D = distance_matrix(X,X,funct)

        # k is the number of clusters
        k = self.n_clusters
        # initialize by randomly sampling k data points
        if(self.initialization == "random"):
            mediod_indices = random_mediod_initalization(self.D, self.n_init, k)

        tightness = []

        # distances to current mediods
        D_m = self.D[np.ix_(mediod_indices)].T
        # assignment to each cluster
        I_assign = np.argmin(D_m,axis=1)
        # the ith index of "clusters" denotes the indices for data points in cluster
        i
        clusters = [np.where(I_assign == i)[0] for i in range(k)]

        # overall tightness
        Q = overall_tightness(D_m)
        tightness.append(Q)
        # collapse the square distance matrix to a vector by summing its rows
        sums = [np.sum(self.D[np.ix_(clusters[i])], axis=0) for i in range(k)]
        # set change to infinity

        change = np.inf
        # run until the tightness does not change or max iterations is exceeded
        while(change > self.tau and self.n_iter_ < self.max_iter):
            # store the 'previous' tightness
            tightness1 = Q
            # take the minimum "collapsed sum" for each data point
            mediod_indices = [np.argmin(sums[i]) for i in range(k)]
            # index slicing of precalculated distance matrix
            D_m = self.D[np.ix_(mediod_indices)].T
            # assignment to each cluster
            I_assign = np.argmin(D_m,axis=1)

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