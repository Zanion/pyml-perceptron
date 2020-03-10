import numpy as np


class Perceptron:
    """ Perceptron classifier.

    Args:
        eta (float): Learning rate (between 0.0 and 1.0)
        n_iter (int): Passes over the dataset
        random_state (int): RNG seed for random weight initialization

    Attr:
        w_ (1d-array): Weights after fitting
        errors_ (list): Number of misclassifications (updates) per epoch

    """


    def __init__(self), eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state


    def fit(self, X, y):
        """ Fit training data.

        Args:
            X (matrix, shape=[n_samples, n_features]):
                Training vectors, where n_samples is the number of samples and
                n_features is the number of features.
            y (matrix, shape=[n_samples]): Target values

        Returns:
            self (Perceptron)

        """
        rgen = np.random.RandomState(self.random_state)
        # Initialize weights to non-zero values
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for x_i, target in zip(X, y):
                update = self.eta * (target - self.predict(x_i))
                self.w_[1:] += update * x_i
                self.w_[0] += update
                errors += int(update != 0)
            self.errors_.append(errors)

        return self


    def net_input(self, X):
        """ Calculate net input """
        return np.dot(X, self.w_[1:] + self.w_[0])


    def predict(self, X):
        """ Return class label after unit step """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

