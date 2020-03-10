import numpy as np


class AdalineSGD:
    """ Adaptive Linear Neuron Classifier (Adaline).

    Args:
        eta (float): Learning rate (between 0.0 and 1.0)
        n_iter (int): Passes over the dataset
        shuffle (bool): Shuffles training data every epoch if True to
            prevent cycles.
        random_state (int): RNG seed for random weight initialization

    Attr:
        w_ (1d-array): Weights after fitting
        cost_ (list): Sum-of-squares cost function value averaged over all
            training samples in each epoch

    """


    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state = random_state


    def fit(self, X, y):
        """ Fit training data.

        Args:
            X (matrix, shape=[n_samples, n_features]):
                Training vectors, where n_samples is the number of samples and
                n_features is the number of features.
            y (matrix, shape=[n_samples]): Target values

        Returns:
            self (AdalineSGD)

        """
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for x_i, target in zip(X, y):
                cost.append(self._update_weights(x_i, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self


    def partial_fit(self, X, y):
        """ Fit training data without reinitializing the weights """
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for x_i, target in zip(X, y):
                self._update_weights(x_i, target)
        else:
            self._update_weights(X, y)
        return self


    def _shuffle(self, X, y):
        """ Shuffle training data """
        self.rgen = np.random.RandomState(self.random_state)
        r = self.rgen.permutation(len(y))
        return X[r], y[r]


    def _initialize_weights(self, m):
        """ Initialize weights to small random numbers """
        rgen = np.random.RandomState(self.random_state)
        # Initialize weights to non-zero values
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True


    def _update_weights(self, x_i, target):
        """ Apply Adaline learning rule to update the weights """
        output = self.activation(self.net_input(x_i))
        error = (target - output)
        self.w_[1:] += self.eta * x_i.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error**2
        return cost


    def net_input(self, X):
        """ Calculate net input """
        return np.dot(X, self.w_[1:]) + self.w_[0]


    def activation(self, X):
        """ Compute linear activation """
        # Use identity of X
        return X


    def predict(self, X):
        """ Return class label after unit step """
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)

