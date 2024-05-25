import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml


def get_data_openml(name):
    ''' get dataset from openml
    Args:
        name: the name of dataset

    Return:
        data: the set of samples
        labels: target values

    '''
    datasets = {
        'cpu': 761,
        'ups': 43947,
        'ipms': 44394,
        'hlc': 45026,
        'hggs': 44419,
        'bnk32': 833,
        'airl': 44528,
        'emv': 43976,
        'cmps': 44462,
        'jnns': 44429,
    }

    dataset = fetch_openml(as_frame=True, data_id=datasets[name], parser="pandas")

    labels_train = dataset.target
    df = dataset.data

    for col_name in df.columns:
        if str(df[col_name].dtype) == 'category' or str(df[col_name].dtype) == 'object':
            df[col_name] = df[col_name].astype('category')
            df[col_name] = df[col_name].cat.codes

    labelencoder = LabelEncoder()
    labels = labelencoder.fit_transform(labels_train)

    data = df.to_numpy()

    print('\nDataset: ', name)
    print('The number of classes: ', len(np.unique(labels, return_counts=True)[0]))
    samples_per_class = [f'class {i}: {samples}' for i,
                         samples in enumerate(np.unique(labels, return_counts=True)[1])]
    print('The number of samples per class: \n', ', '.join(samples_per_class), '\n')

    return data, labels
