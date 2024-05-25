import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import FitFailedWarning
from pandas.errors import SettingWithCopyWarning


from utils.al import Sampling
from utils.rlda import RLDA

from utils.helpers import hyperparams_tuning
from utils.helpers import get_model_and_params_grid
from utils.helpers import print_table
from utils.helpers import train_scarf

from utils.datasets import get_data_openml

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset', type=str, default='airl',
    choices=['cpu', 'ups', 'hlc', 'hggs', 'ipms', 'bnk32', 'cmps', 'airl', 'emv', 'jnns'],
    help='name of the dataset')
parser.add_argument('--label_size', type=int,
                    default=100, help='size of labeled set')
parser.add_argument('--budget_size', type=int,
                    default=10, help='budget size')
parser.add_argument('--test_size', type=float,
                    default=0.3, help='test set proportion rangin from 0.1 to 1.0')
parser.add_argument('--cycles', type=int,
                    default=20, help='active learning cycles')
parser.add_argument('--model', type=str,  default='RLDA',
                    choices=['RLDA', 'LRR', 'LSVM', 'MLP', 'RF'], help='classifier')
parser.add_argument(
    '--sampling', type=str, default='ALRLDA',
    choices=['ALRLDA', 'random', 'uncertainty', 'diversity', 'ProbCover', 'BALD'],
    help='sampling method')

parser.add_argument('--SS', type=int,
                    default=20, help='number of random shuffle and splits')


opt = parser.parse_args()


def main():
    np.random.seed(42)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=FitFailedWarning)

    print('hello')

    split_test_accs = []

    dataset_name = opt.dataset
    data, labels = get_data_openml(dataset_name)

    n_splits = opt.SS

    test_size = opt.test_size
    label_size = opt.label_size
    budget_size = opt.budget_size

    cycles = opt.cycles

    model_name = opt.model

    sampling_method = opt.sampling

    al_sampling = Sampling(budget_size)

    model, param_grid = get_model_and_params_grid(model_name, data.shape)

    count_splits = 0
    print(f'Running ...')

    sss = StratifiedShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=42)
    for train_unl_index, test_index in sss.split(data, labels):
        X_unl_train, X_test = data[train_unl_index], data[test_index]
        y_unl_train, y_test = labels[train_unl_index], labels[test_index]

        X_lab, X_unl, y_lab, y_unl = train_test_split(
            X_unl_train, y_unl_train, train_size=label_size, random_state=42, stratify=y_unl_train)

        count_splits += 1
        cycle_accs = []

        # tuning gamma for RLDA and ALRLDA
        param_grid_rlda = {"clf__gamma": [0.1, 1, 10, 100]}
        best_params_rlda, best_model_rlda = hyperparams_tuning(
            RLDA(), param_grid_rlda, X_lab, y_lab)
        gamma = best_params_rlda['clf__gamma']

        scaler = StandardScaler()
        scaler.fit(X_lab)
        X_lab_scaled = scaler.transform(X_lab)
        X_unl_scaled = scaler.transform(X_unl)
        X_test_scaled = scaler.transform(X_test)

        # tuning hyperparams for the rest of the models
        if model_name != 'RLDA':
            _, best_model = hyperparams_tuning(model, param_grid, X_lab_scaled, y_lab)
            test_pred = best_model.predict(X_test_scaled)

        if model_name == 'RLDA':
            test_pred = best_model_rlda.predict(X_test)

        test_acc = accuracy_score(y_true=y_test, y_pred=test_pred)
        cycle_accs.append(test_acc)

        X_train_scaled = np.copy(X_lab_scaled)
        X_train = np.copy(X_lab)
        y_train = np.copy(y_lab)

        if sampling_method == 'ProbCover':
            scarf_model = train_scarf(X_lab_scaled, X_unl_scaled)

        for i_cycle in range(cycles):
            pred_probas = best_model_rlda.predict_proba(X_unl)
            if sampling_method == 'ALRLDA':
                active_idx = al_sampling.ALRLDA(X_train, y_train, X_unl, gamma)
            if sampling_method == 'random':
                active_idx = al_sampling.random(X_unl)
            if sampling_method == 'uncertainty':
                active_idx = al_sampling.uncertainty(pred_probas)
            if sampling_method == 'diversity':
                active_idx = al_sampling.diversity(pred_probas)
            if sampling_method == 'BALD':
                active_idx = al_sampling.BALD(X_unl, pred_probas)
            if sampling_method == 'ProbCover':
                active_idx = al_sampling.ProbCover(X_train_scaled, X_unl_scaled, scarf_model)

            X_unl_opt_scaled = X_unl_scaled[active_idx]
            X_unl_opt = X_unl[active_idx]
            y_unl_opt = y_unl[active_idx]

            rest_idx = np.array([i for i in range(len(X_unl)) if i not in active_idx])

            X_unl_scaled = X_unl_scaled[rest_idx]
            X_unl = X_unl[rest_idx]
            y_unl = y_unl[rest_idx]

            X_train_scaled = np.concatenate((X_train_scaled, X_unl_opt_scaled))
            X_train = np.concatenate((X_train, X_unl_opt))
            y_train = np.concatenate((y_train, y_unl_opt))

            if model_name != 'RLDA':
                best_model.fit(X_train_scaled, y_train)
                test_pred = best_model.predict(X_test_scaled)

            best_model_rlda.fit(X_train, y_train)
            if model_name == 'RLDA':
                test_pred = best_model_rlda.predict(X_test)
            test_acc = accuracy_score(y_true=y_test, y_pred=test_pred)
            cycle_accs.append(test_acc)

        split_test_accs.append(cycle_accs)

        print(f'splits completed: {count_splits} / {n_splits}')

    avg_accs_over_splits = np.mean(split_test_accs, axis=0)
    avg_stds_over_splits = np.std(split_test_accs, axis=0)

    print('\nResult: ')
    print_table(avg_accs_over_splits, avg_stds_over_splits)


if __name__ == '__main__':
    main()
