'''
K-Nearest-Neighbour implementation
'''
import math

class KNN:

    _k = 1
    _distance_calc = None
    _X = None
    _y = None

    def __init__(self, k=2, distance_metric="euclidean"):
        self._set_k(k)
        self._set_distance_metric(distance_metric)

    def fit(self, X, y):
        self._X = X
        self._y = y

    def predict(self, X):
        prediction = []
        for i, row_test in enumerate(X):
            distance = []
            rows = []
            for j, row_train in enumerate(self._X):
                distance.append((j, self._distance_calc(row_train, row_test)))
            for j, dist in sorted(distance, key=lambda t: t[1])[0:self._k]:
                rows.append(self._y[j])
            prediction.append(self._mean(rows))
        return prediction

    def _euclidian_distance(self, row_train, row_test):
        dist = 0
        for i in range(0, len(row_train)):
            dist += (row_train[i] - row_test[i]) * (row_train[i] - row_test[i]) 
        return math.sqrt(dist)

    def _manhattan_distance(self, row_train, row_test):
        dist = 0 
        for i in range(0, len(row_train)):
            dist += abs(row_train[i] - row_test[i])
        return dist

    def _mean(self, arr):
        return float(sum(arr)) / max(len(arr), 1)

    '''
    SETTERS 
    '''
    def _set_k(self, k):
        self._k = 1 if k <= 0 else k

    def _set_distance_metric(self, distance_metric):
        if distance_metric == "euclidean":
            self._distance_calc = self._euclidian_distance
        elif distance_metric == "manhattan":
            self._distance_calc = self._manhattan_distance 
        else:
            print(f"{distance_metric} is not supported, using euclidean")
            self._distance_metric = self._euclidian_distance


