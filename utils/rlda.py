import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class RLDA(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=1):
        self.gamma = gamma
        self.p = 0

        self.H_rlda = 0
        self.sample_mean_0_rlda = 0
        self.sample_mean_1_rlda = 0
        self.alpha_0_rlda = 0.5
        self.alpha_1_rlda = 0.5

    def fit(self, X, y):
        sample_mean_0 = np.mean(X[y == 0], axis=0).reshape(-1, 1)
        sample_mean_1 = np.mean(X[y == 1], axis=0).reshape(-1, 1)

        size_0 = np.count_nonzero(y == 0)
        size_1 = np.count_nonzero(y == 1)

        p = X.shape[1]

        X_0 = X[y == 0]
        X_1 = X[y == 1]

        cov_0 = np.cov(X_0, rowvar=False)
        cov_1 = np.cov(X_1, rowvar=False)

        sample_pooled_cov = ((size_0 - 1)*cov_0 + (size_1 - 1)*cov_1) / (size_0 + size_1 - 2)

        H_term_1 = (np.identity(p) + self.gamma * sample_pooled_cov)
        H = np.linalg.inv(H_term_1)

        self.H_rlda = H
        self.sample_mean_0_rlda = sample_mean_0
        self.sample_mean_1_rlda = sample_mean_1

        self.alpha_0_rlda = size_0 / X.shape[0]
        self.alpha_1_rlda = size_1 / X.shape[0]

    def predict(self, X_test):
        y_pred = []
        for i in range(X_test.shape[0]):
            y_pred_i = self.RLDA_classifier(X_test[i])
            y_pred.append(y_pred_i)

        return y_pred

    def RLDA_classifier(self, x):
        x = x.reshape(-1, 1)
        term_1 = (x - (self.sample_mean_0_rlda + self.sample_mean_1_rlda)/2).T
        term_2 = np.matmul(term_1,  self.H_rlda)
        term_3 = np.matmul(term_2, (self.sample_mean_0_rlda - self.sample_mean_1_rlda))
        term_4 = self.gamma * term_3
        W_RLDA = term_4[0, 0] - np.log(self.alpha_1_rlda / self.alpha_0_rlda)

        if W_RLDA > 0:
            return 0
        else:
            return 1

    def predict_proba(self, X_test):
        y_probas = []
        for i in range(X_test.shape[0]):
            probas = self.discriminant_val(X_test[i])
            y_probas.append(probas)
        return y_probas

    def discriminant_val(self, x):
        x = x.reshape(-1, 1)

        term1 = x - self.sample_mean_0_rlda
        term2 = np.matmul(self.H_rlda, term1)
        term3 = np.matmul(term1.T, term2)
        discriminant_val_0 = self.gamma * term3 * (-1/2) + np.log(self.alpha_0_rlda)

        term11 = x - self.sample_mean_1_rlda
        term22 = np.matmul(self.H_rlda, term11)
        term33 = np.matmul(term11.T, term22)
        discriminant_val_1 = self.gamma * term33 * (-1/2) + np.log(self.alpha_1_rlda)

        discriminant_values = [discriminant_val_0[0, 0], discriminant_val_1[0, 0]]
        probas = self.softmax(discriminant_values)
        return probas

    def softmax(self, x):
        term1 = np.exp(x - np.max(x))
        res = term1 / term1.sum(axis=0)
        return res
