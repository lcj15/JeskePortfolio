from Classifier import Classifier as Classifier



def alpha(t, B, a0=0.9):
    """
    Calculates the learning rate that determines the step size at time t
    :param t: current iteration number or "time"
    :param B: constant parameter
    :param a0: constant parameter
    :return: learning rate at time t
    """
    return a0 * np.exp(-1 * B * t)

# TODO: Add comments


class LVQ(Classifier):
    """
    Implementation of Learning Vector Quantization in Python.
    """

    def __init__(self, L=5, N=1000, initialization="class conditional", offset=0.02):
        """
        Initialize Learning Vector Quantization model
        :param L: number of prototypes
        :param N: number of iterations
        :param initialization: class conditional samples around class cluster means with small
        random offset, otherwise randomly samples data points as initial prototypes
        """
        self.prototype_indices = None
        self.flattened_prototypes = None
        self.k = None
        self.prototypes = None
        self.L = L
        self.N = N
        self.t = 0
        self.initialization = initialization
        self.offset = offset

    def fit(self, X, y):
        """
        Fit the model to the data
        :param X: training data
        :param y: training labels
        :return: None
        """


        # data info
        self.num_samples, self.num_features = X.shape
        k = len(np.unique(y))
        # data partitioned by class
        class_indices = [np.where(y == i)[0] for i in range(k)]
        class_data = [X[i] for i in class_indices]
        class_cardinality = [len(class_indices[i]) for i in range(k)]

        # list of class means, the ith entry is the mean vector for the ith class
        if self.initialization == "class conditional":
            class_means = [np.mean(i, axis=0) for i in class_data]
            # normal random initialization of prototypes with class means and class variance

            # creates the prototype matrix
            # 0.01 is a small random offset for initialization
            prototypes = [np.random.normal(a, self.offset, size=(self.L, self.num_features)) for a in class_means]

        else:
            prototypes = []
            for i in range(k):
                ii = np.random.choice(class_cardinality[i], self.L, replace=False)
                prototypes.append(X[np.ix_(ii)])

        prototype_indices = np.concatenate([np.array([i] * self.L) for i in range(k)])
        flattened_prototypes = np.concatenate(prototypes)

        while self.t < self.N:
            ii = np.random.choice(len(X), replace=True)
            samp = X[ii]
            samp_lab = y[ii]
            dm = distance_matrix(flattened_prototypes, [samp])
            bmu_i = np.argmin(dm)
            m_star = flattened_prototypes[bmu_i]
            move = alpha(self.t, B=np.log10(10) / self.N) * np.subtract(samp, m_star)

            if samp_lab == prototype_indices[bmu_i]:
                m_star += move
            else:
                m_star -= move
            self.t += 1

        # record keeping
        self.k = k
        self.prototypes = flattened_prototypes.reshape(self.L, self.k, self.num_features)
        self.flattened_prototypes = flattened_prototypes
        self.prototype_indices = prototype_indices

    def predict(self, X):
        """
        Make predictions on unseen data
        :param X: unseen data
        :return: labels corresponding to class predictions
        """
        dm = distance_matrix(self.flattened_prototypes, X)
        return self.prototype_indices[np.ix_(np.argmin(dm, axis=0))]

    def plot_prototypes(self):
        """
        Plot prototypes.  Intended for ECG dataset.  WIll not be applicable to most datasets.

        :return: None
        """
        fig, ax = plt.subplots(figsize=(self.L * 10, self.L * 2), nrows=self.k, ncols=self.L)

        titles = ["Normal ECG", "Abnormal ECG"]
        colors = ["red", "blue"]
        counter = 0
        for i in range(self.k):
            for j in range(self.L):
                ax[i][j].plot(self.flattened_prototypes[counter], c=colors[i], label=titles[i])
                counter += 1

        ax[0][0].legend()
        ax[1][0].legend()
        fig.suptitle("Prototypes Plotted as ECG's", size=self.L * 12)
