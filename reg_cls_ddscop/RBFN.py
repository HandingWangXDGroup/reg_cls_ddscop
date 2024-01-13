# Radial basis function network

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score


class RBFN(object):
    def __init__(self, num_neurons, kernel):
        self.num_neurons = num_neurons
        self.sigma = None
        self.centers = None
        self.weights = None
        self.bias = None



    def kernel_(self, data_point):  # Gaussian function
        G = np.zeros((data_point.shape[0], self.num_neurons))
        for i in range(self.num_neurons):
            G[:, i] = np.sum(np.exp(-0.5 * ((data_point - self.centers[i, :]) / self.sigma) ** 2), axis=1)
        return G

    def calsigma(self):
        max = 0.0
        num = 0
        total = 0.0
        for i in range(self.num_neurons - 1):
            for j in range(i + 1, self.num_neurons):
                dis = np.linalg.norm(self.centers[i] - self.centers[j])
                total = total + dis
                num += 1
                if dis > max:
                    max = dis
        self.sigma = 2 * total / num

    def fit(self, X, Y):
        km = KMeans(n_clusters=self.num_neurons).fit(X)
        self.centers = km.cluster_centers_
        self.calsigma()
        G = self.kernel_(X)
        temp = np.column_stack((G, np.ones((X.shape[0]))))
        temp = np.dot(np.linalg.pinv(temp), Y)
        self.weights = temp[:self.num_neurons]
        self.bias = temp[self.num_neurons]

    def predict(self, X):
        X = np.array(X)
        G = self.kernel_(X)
        predictions = np.dot(G, self.weights) + self.bias
        return predictions

    def score(self, X, Y):
        # MSE = np.sum((y_te - y_pred) ** 2) / len(y_te)
        # R2 = 1 - MSE / np.var(y_te)
        return r2_score(Y.reshape(-1, 1), self.predict(X))


'''
        def Reflectedfun( data_point):  # Reflected function
            G = np.zeros((data_point.shape[0], num_neurons))
            for i in range(num_neurons):
                G[:, i] = np.sum(1 / (1 + np.exp(((data_point - self.centers[i, :]) / self.sigma) ** 2)), axis = 1)
            return G

'''