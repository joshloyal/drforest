import numpy as np
import openml
import pandas as pd
import scipy.sparse as sp


__all__ = ['query_regression_tasks', 'load_openml']


def query_regression_tasks(n_samples_min=100, n_samples_max=5000,
                           n_features_max=None):
    task_list = openml.tasks.list_tasks(task_type_id=2)
    tasks = pd.DataFrame.from_dict(task_list, orient='index')

    # filter tasks with all numeric features
    tasks = (tasks.query('NumberOfInstances <= {}'.format(n_samples_max))
                  .query('NumberOfInstances >= {}'.format(n_samples_min))
                  #.query('NumberOfNumericFeatures == NumberOfFeatures')
                  .query('NumberOfInstancesWithMissingValues == 0.0'))

    if n_features_max:
        tasks = tasks.query('NumberOfFeatures <= {}'.format(n_features_max))

    return tasks[['did', 'name', 'NumberOfInstances', 'NumberOfFeatures']]


def load_openml(name, n_samples_min=100, n_samples_max=50000):
    tasks = query_regression_tasks(n_samples_min=n_samples_min,
                                   n_samples_max=n_samples_max)

    dataset_id = int(tasks.query('name == "{0}"'.format(name))['did'].values[0])

    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format='array',
            target=dataset.default_target_attribute
    )

    if sp.issparse(X):
        X = X.toarray()

    return X.astype(np.float64), y.astype(np.float64)
