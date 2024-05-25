import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPClassifier as MLP
from sklearn.linear_model import LogisticRegression as LRR
from sklearn.svm import LinearSVC as LSVM
from sklearn.ensemble import RandomForestClassifier as RF
from utils.rlda import RLDA

from torch.optim import Adam
from torch.utils.data import DataLoader

from utils.scarf.utils import get_device, dataset_embeddings, fix_seed, train_epoch

from utils.scarf.loss import NTXent
from utils.scarf.model import SCARF
from utils.scarf.dataset import SCARFDataset


def get_model_and_params_grid(model_name, data_shape):

    if model_name == 'RLDA':
        model = RLDA()
        param_grid = {
            "clf__gamma": [0.1, 1, 10, 100],
        }
    elif model_name == 'LRR':
        model = LRR(random_state=42)
        param_grid = {
            'clf__penalty': ['l2'],
            'clf__C': [100, 10, 1.0, 0.1, 0.01]
        }
    elif model_name == 'LSVM':
        model = LSVM(random_state=42)
        param_grid = {
            'clf__penalty': ['l1', 'l2'],
            'clf__C': [0.1, 0.5, 1, 5, 10],
            'clf__loss': ['hinge', 'squared_hinge'],
        }
    elif model_name == 'MLP':
        hidden_layer = (int(data_shape[1]/2), )
        model = MLP(random_state=42, hidden_layer_sizes=hidden_layer)
        param_grid = {
            "clf__solver": ['lbfgs', 'adam', 'sgd'],
            "clf__activation": ['logistic', 'relu'],
        }

    elif model_name == 'RF':
        model = RF(random_state=42)
        param_grid = {
            'clf__n_estimators': [1, 5, 10, 20],
            'clf__max_depth': [2, 5, 10,],
            'clf__max_features': ['sqrt', 'log2']
        }

    return model, param_grid


def hyperparams_tuning(model, param_grid, X_train, y_train):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline(steps=[("clf", model)])

    search = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=1)
    search.fit(X_train, y_train)

    return search.best_params_, search.best_estimator_


def print_table(avg_accs_over_splits, avg_stds_over_splits):
    num_columns = 3
    data = zip(np.arange(22), [round(acc, 4) for acc in avg_accs_over_splits], [
        round(std, 4) for std in avg_stds_over_splits])

    col_width = len('AL cycle')
    print("|".join(str(item).ljust(col_width) for item in ['AL cycle', 'Test acc', 'std']))
    print("+".join("-" * col_width for _ in range(num_columns)))
    for i, row in enumerate(data):
        print("|".join(str(item).ljust(col_width) for item in row))


def split_bugdet(budget_size):
    if budget_size % 2 != 0:
        half1 = (budget_size + 1) // 2
        half2 = budget_size // 2
    else:
        half1 = budget_size // 2
        half2 = budget_size // 2
    return half1, half2


def predict_prob_dropout_split(X_unl, predict_proba, n_drop):
    num_class = 2
    probs = np.zeros((n_drop, X_unl.shape[0], num_class))
    for i in range(n_drop):
        for idxs, out in enumerate(predict_proba):
            probs[i][idxs] += out
    return probs


def train_scarf(X_lab_scaled, X_unl_scaled):
    X_features = np.concatenate((X_lab_scaled, X_unl_scaled))
    cc = np.all(X_features[1:] == X_features[:-1], axis=0)
    X_features = X_features[:, ~cc]
    train_ds = SCARFDataset(X_features)

    batch_size = 128
    epochs = 1000
    device = get_device()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = SCARF(
        input_dim=train_ds.shape[1],
        features_low=train_ds.features_low,
        features_high=train_ds.features_high,
        dim_hidden_encoder=256,
        num_hidden_encoder=4,
        dim_hidden_head=256,
        num_hidden_head=2,
        corruption_rate=0.6,
        dropout=0.1,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    ntxent_loss = NTXent()

    loss_history = []

    for epoch in range(1, epochs + 1):
        epoch_loss = train_epoch(model, ntxent_loss, train_loader, optimizer, device)
        loss_history.append(epoch_loss)

        if epoch % 10 == 0:
            print(f"epoch {epoch}/{epochs} - loss: {loss_history[-1]:.4f}", end="\r")

    return model


def get_emb(model, train_ds):
    batch_size = 128
    device = get_device()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    train_embeddings = dataset_embeddings(model, train_loader, device)

    return train_embeddings
