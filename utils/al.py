import numpy as np
import pandas as pd
import torch

from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import norm
from utils.helpers import split_bugdet
from utils.helpers import predict_prob_dropout_split
from utils.helpers import get_emb
from utils.scarf.dataset import SCARFDataset


class ALRLDA(BaseEstimator, ClassifierMixin):
    '''
    Args:
        gamma: a regularization parameter
    '''

    def __init__(self, gamma=1):
        self.sample_mean_0 = 0
        self.sample_mean_1 = 0
        self.size_0 = 0
        self.size_1 = 0
        self.alpha_0 = 0.5
        self.alpha_1 = 0.5

        self.gamma = gamma
        self.p = 0
        self.H = 0

        self.sample_mean_0_L = 0
        self.sample_mean_1_L = 0
        self.size_0_L = 0
        self.size_1_L = 0
        self.K_L = 0
        self.sample_pooled_cov_L = 0

    def fit_once(self, X_lab, y_lab):
        '''
        Args:
            X_lab: labeled data
            y_lab: target values

        '''
        self.p = X_lab.shape[1]

        self.sample_mean_0_L = np.mean(X_lab[y_lab == 0], axis=0).reshape(-1, 1)
        self.sample_mean_1_L = np.mean(X_lab[y_lab == 1], axis=0).reshape(-1, 1)

        self.size_0_L = np.count_nonzero(y_lab == 0)
        self.size_1_L = np.count_nonzero(y_lab == 1)

        self.alpha_0 = self.size_0_L / X_lab.shape[0]
        self.alpha_1 = self.size_1_L / X_lab.shape[0]

        X_0 = X_lab[y_lab == 0]
        X_1 = X_lab[y_lab == 1]

        cov_0 = np.cov(X_0, rowvar=False)
        cov_1 = np.cov(X_1, rowvar=False)

        sample_pooled_cov = ((self.size_0_L - 1)*cov_0 + (self.size_1_L - 1)
                             * cov_1) / (self.size_0_L + self.size_1_L - 2)
        self.sample_pooled_cov_L = sample_pooled_cov

        a_res = self.kappa(self.size_0_L+self.size_1_L-1)

        H = (np.identity(self.p) + a_res * self.gamma * sample_pooled_cov)
        self.K_L = np.linalg.inv(H)

    def fit_error(self, x_unl):
        '''
        Args:
            x_unl: unlabeled data point

        Return:
            estimated error
        '''

        # if x_unl goes to 0
        err_hat_x_unl_to_0 = self.fit_x_unl(x_unl, 0)

        # if x_unl goes to 1
        err_hat_x_unl_to_1 = self.fit_x_unl(x_unl, 1)

        err_hat_x_unl = self.alpha_0 * err_hat_x_unl_to_0 + self.alpha_1 * err_hat_x_unl_to_1

        return err_hat_x_unl

    def fit_x_unl(self, x_unl, i):
        '''
        Args:
            x_unl: unlabeled data point
            i: class label i

        Return:
            estimated error (assuming x_unl assigned to class i)
        '''
        if i == 0:
            self.sample_mean_0 = self.sample_mean_0_L + self.d_func(
                x_unl, self.sample_mean_0_L) / (self.size_0_L + 1)
            self.sample_mean_1 = self.sample_mean_1_L

            self.H = self.upd_H(x_unl, self.size_0_L, self.sample_mean_0_L)

            self.size_0 = self.size_0_L + 1
            self.size_1 = self.size_1_L

            sample_pooled_cov = self.upd_sample_pooled_cov(
                x_unl, self.size_0_L, self.sample_mean_0_L)

        if i == 1:
            self.sample_mean_0 = self.sample_mean_0_L
            self.sample_mean_1 = self.sample_mean_1_L + self.d_func(
                x_unl, self.sample_mean_1_L) / (self.size_1_L + 1)

            self.H = self.upd_H(x_unl, self.size_1_L, self.sample_mean_1_L)

            self.size_0 = self.size_0_L
            self.size_1 = self.size_1_L + 1

            sample_pooled_cov = self.upd_sample_pooled_cov(
                x_unl, self.size_1_L, self.sample_mean_1_L)

        G_0 = self.G(self.sample_mean_0)[0, 0]
        G_1 = self.G(self.sample_mean_1)[0, 0]

        delta_hat = self.delta_hat()

        D = self.D(sample_pooled_cov)[0, 0]

        G_term_0 = (-1)**(0+1) * (self.size_0 + self.size_1 - 2) * delta_hat / self.size_0
        G_term_1 = (-1)**(1+1) * (self.size_0 + self.size_1 - 2) * delta_hat / self.size_1

        G_term_alpha = np.log(self.alpha_1 / self.alpha_0) / self.gamma

        G_hat_0 = G_0 + G_term_0 - G_term_alpha
        G_hat_1 = G_1 + G_term_1 - G_term_alpha

        D_hat = ((1 + self.gamma * delta_hat)**2) * D

        err_hat_0 = self.err_hat(G_hat_0, D_hat, 0)
        err_hat_1 = self.err_hat(G_hat_1, D_hat, 1)

        err_hat = self.alpha_0 * err_hat_0 + self.alpha_1 * err_hat_1

        return err_hat

    def G(self, sample_mean):
        term_1 = (sample_mean - (self.sample_mean_0 + self.sample_mean_1)/2).T
        term_2 = np.matmul(term_1, self.H)
        res = np.matmul(term_2, (self.sample_mean_0 - self.sample_mean_1))

        return res

    def D(self, sample_cov):
        term_1 = self.sample_mean_0 - self.sample_mean_1
        term_2 = np.matmul(term_1.T, self.H)
        term_3 = np.matmul(term_2, sample_cov)
        term_4 = np.matmul(term_3, self.H)
        res = np.matmul(term_4, term_1)
        return res

    def delta_hat(self):
        term_1 = self.p / (self.size_0 + self.size_1 - 2)
        term_2 = np.trace(self.H) / (self.size_0 + self.size_1 - 2)
        term_3 = term_1 - term_2

        term_4 = 1 - term_1 + term_2

        res = term_3 / (self.gamma * term_4)

        return res

    def err_hat(self, G_hat, D_hat, i):
        term_1 = G_hat * (-1)**(i+1) / np.sqrt(D_hat)
        res = norm.cdf(term_1)

        return res

    def upd_sample_pooled_cov(self, x_new, size_i, sample_mean_i):
        a_res = self.kappa(self.size_0_L+self.size_1_L-1)
        term1 = a_res * self.sample_pooled_cov_L

        beta_res = self.beta(size_i, self.size_0_L + self.size_1_L - 1)
        term2 = beta_res * self.D_func(x_new, sample_mean_i)

        sample_pooled_cov_upd = term1 + term2

        return sample_pooled_cov_upd

    def upd_H(self, x_new, size_i, sample_mean_i):
        beta_res = self.beta(size_i, self.size_0_L + self.size_1_L - 1)

        term2 = beta_res * self.D_func(x_new, sample_mean_i) * self.gamma
        term3 = np.matmul(term2, self.K_L)

        d_res = self.d_func(x_new, sample_mean_i)

        term4 = beta_res * d_res.T
        term5 = np.matmul(term4, self.K_L)
        term6 = np.matmul(term5, d_res) * self.gamma + 1

        term7 = term3 / term6[0, 0]
        term8 = np.identity(self.p) - term7

        H_upd = np.matmul(self.K_L, term8)

        return H_upd

    def d_func(self, x_new, sample_mean):
        x_new_reshape = x_new.reshape(-1, 1)
        res = x_new_reshape - sample_mean
        return res

    def D_func(self, x_new, sample_mean):
        res = self.d_func(x_new, sample_mean)
        res2 = np.matmul(res, res.T)
        return res2

    def kappa(self, n):
        res = (n - 1) / n
        return res

    def beta(self, n0, n):
        res = n0 / (n*(n0+1))
        return res


class ProbCover:
    def __init__(self, X_features, lSet, uSet, budgetSize, delta):
        self.seed = 42
        self.all_features = X_features,
        self.lSet = lSet
        self.uSet = uSet
        self.budgetSize = budgetSize
        self.delta = delta
        self.relevant_indices = 0
        self.rel_features = 0
        self.graph_df = 0

    def construct_graph(self, batch_size=100):
        """
        creates a directed graph where:
        x->y iff l2(x,y) < delta.

        represented by a list of edges (a sparse matrix).
        stored in a dataframe
        """
        xs, ys, ds = [], [], []
        print(f'Start constructing graph using delta={self.delta}')

        # distance computations are done in GPU
        cuda_feats = torch.tensor(self.rel_features).cuda()
        # print(len(self.rel_features))
        for i in range(len(self.rel_features) // batch_size):
            # distance comparisons are done in batches to reduce memory consumption
            cur_feats = cuda_feats[i * batch_size: (i + 1) * batch_size]
            dist = torch.cdist(cur_feats, cuda_feats)
            mask = dist < self.delta
            # saving edges using indices list - saves memory.
            x, y = mask.nonzero().T
            xs.append(x.cpu() + batch_size * i)
            ys.append(y.cpu())
            ds.append(dist[mask].cpu())

        xs = torch.cat(xs).numpy()
        ys = torch.cat(ys).numpy()
        ds = torch.cat(ds).numpy()

        # print(xs.shape)
        df = pd.DataFrame({'x': xs, 'y': ys, 'd': ds})
        print(f'Finished constructing graph using delta={self.delta}')
        print(f'Graph contains {len(df)} edges.')
        return df

    def select_samples(self):
        """
        selecting samples using the greedy algorithm.
        iteratively:
        - removes incoming edges to all covered samples
        - selects the sample high the highest out degree (covers most new samples)

        """
        print(self.all_features)
        print(self.all_features[0].shape)
        self.relevant_indices = np.concatenate([self.lSet, self.uSet])
        self.rel_features = self.all_features[0][self.relevant_indices]

        self.graph_df = self.construct_graph()

        print(f'Start selecting {self.budgetSize} samples.')
        selected = set()
        # removing incoming edges to all covered samples from the existing labeled set
        edge_from_seen = np.isin(self.graph_df.x, np.arange(len(self.lSet)))
        covered_samples = self.graph_df.y[edge_from_seen].unique()
        cur_df = self.graph_df[(~np.isin(self.graph_df.y, covered_samples))]
        for i in range(self.budgetSize):
            coverage = len(covered_samples) / len(self.relevant_indices)
            # selecting the sample with the highest degree
            degrees = np.bincount(cur_df.x, minlength=len(self.relevant_indices))
            print(
                f'Iteration is {i}.\tGraph has {len(cur_df)} edges.\tMax degree is {degrees.max()}.\tCoverage is {coverage:.3f}')
            cur = degrees.argmax()

            # removing incoming edges to newly covered samples
            new_covered_samples = cur_df.y[(cur_df.x == cur)].values
            assert len(np.intersect1d(covered_samples, new_covered_samples)
                       ) == 0, 'all samples should be new'
            cur_df = cur_df[(~np.isin(cur_df.y, new_covered_samples))]

            covered_samples = np.concatenate([covered_samples, new_covered_samples])
            selected.add(cur)

        selected = list(selected)
        activeSet = self.relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        return activeSet, remainSet


class Sampling:
    """
    Sampling methods which are used to sample
    active learning points from unlabelled set.
    """

    def __init__(self, budget_size):
        self.budget_size = budget_size

    def random(self, X_unl):
        D = np.ones(X_unl.shape[0]) / X_unl.shape[0]
        active_idx = np.random.choice(a=X_unl.shape[0], size=self.budget_size, replace=False, p=D)

        return active_idx

    def uncertainty(self, pred_probas):
        # pred_probas = rlda_model.predict_proba(X_unl)

        probas_max = np.max(pred_probas, axis=1)
        active_idx = np.argpartition(probas_max, self.budget_size)[:self.budget_size]

        return active_idx

    def diversity(self, pred_probas):
        # pred_probas = rlda_model.predict_proba(X_unl)

        probas_max = np.max(pred_probas, axis=1)

        budget_size_1, budget_size_2 = split_bugdet(self.budget_size)

        lowest_idx = np.argpartition(probas_max, budget_size_1)[:budget_size_1]
        highest_idx = np.argpartition(-probas_max, budget_size_2)[:budget_size_2]

        active_idx = np.concatenate((lowest_idx, highest_idx))

        return active_idx

    def BALD(self, X_unl, pred_probas):
        n_drop = 10
        idxs_unlabeled = np.arange(X_unl.shape[0])
        probs = predict_prob_dropout_split(X_unl, pred_probas, n_drop)
        pb = np.mean(probs, axis=0)
        entropy1 = np.sum(-pb*np.log2(pb+1e-6), axis=1)
        entropy2 = np.mean(np.sum(-probs*np.log2(probs+1e-6), axis=2), axis=0)
        U = entropy2 - entropy1

        active_idx = idxs_unlabeled[np.argsort(U)[:self.budget_size]]

        return active_idx

    def ALRLDA(self, X_train, y_train, X_unl, gamma):
        errs = []
        alrlda_sampling = ALRLDA(gamma=gamma)
        alrlda_sampling.fit_once(X_train, y_train)
        for i in range(X_unl.shape[0]):
            err = alrlda_sampling.fit_error(X_unl[i])
            errs.append(err)

        errs = np.array(errs)
        active_idx = np.argpartition(errs, self.budget_size)[:self.budget_size]

        return active_idx

    def ProbCover(self, X_lab, X_unl, scarf_model):
        delta = 0.6
        X_features = np.concatenate((X_lab, X_unl))

        cc = np.all(X_features[1:] == X_features[:-1], axis=0)
        X_features = X_features[:, ~cc]

        train_ds = SCARFDataset(X_features)

        scarf_embeddings = get_emb(scarf_model, train_ds)

        lSet = np.arange(0, len(X_lab), 1)
        uSet = np.arange(len(X_lab), len(X_unl), 1)

        probcov = ProbCover(X_features=scarf_embeddings, lSet=lSet,
                            uSet=uSet, budgetSize=self.budget_size, delta=delta)
        active_idx, _ = probcov.select_samples()

        return active_idx
