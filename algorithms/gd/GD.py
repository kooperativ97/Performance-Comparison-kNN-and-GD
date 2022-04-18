'''
Gradient Decent implementation
'''
import math
import numpy as np

class GD:
    _nepoch = 0
    _learning_rate = 0
    _X = []
    _y = []
    _w0 = 0 # Bias
    _W = [] # Theta

    def __init__(self, nepoch=10000, learning_rate = 0.01):
        self._nepoch = nepoch
        self._learning_rate = learning_rate

    def fit(self, X, y):
        self._X = np.array(X)
        self._y = np.array(y)
        self._w0 = 1

        n_features = len(X[0])
        n_points = len(X)

        self._W = np.ones(n_features)

        cost_history = []

        for i in range(self._nepoch):

            w0_gradient = (2/n_points) * np.sum((self._y - self.prediction()) * (-1)) 
            w_gradient = (2/n_points) * np.sum(np.dot((self._y - self.prediction()) , (-self._X)))

            self._w0 = self._w0 - self._learning_rate * w0_gradient
            self._W = self._W - self._learning_rate * w_gradient


        return cost_history

    def predict(self, X):
        self._X = X
        return self.prediction()

    def prediction(self):
        return np.dot(self._X, self._W) + self._w0

    def mean_squared_error(self):
        error = (self._y - (np.dot(self._X, self._W) + self._w0))**2
        return np.mean(error)


 