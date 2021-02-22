import numpy as np

from sklearn.utils import check_random_state, check_array


__all__ = ['generate_discriminative_dataset']


def bootstrap_sample_column(X, n_samples=None, random_state=1234):
    """bootstrap_sample_column

    Bootstrap sample the column of a dataset.

    Parameters
    ----------
    X : np.ndarray (n_samples,)
        Column to bootstrap

    n_samples : int
        Number of samples to generate. If `None` then generate
        a bootstrap of size of `X`.

    random_state : int
        Seed to the random number generator.

    Returns
    -------
    np.ndarray (n_samples,):
        The bootstrapped column.
    """
    random_state = check_random_state(random_state)
    if n_samples is None:
        n_samples = X.shape[0]

    return random_state.choice(X, size=n_samples, replace=True)


def uniform_sample_column(X, n_samples=None, random_state=1234):
    """uniform_sample_column

    Sample a column uniformly between its minimum and maximum value.

    Parameters
    ----------
    X : np.ndarray (n_samples,)
        Column to sample.

    n_samples : int
        Number of samples to generate. If `None` then generate
        a bootstrap of size of `X`.

    random_state : int
        Seed to the random number generator.

    Returns
    -------
    np.ndarray (n_samples,):
        Uniformly sampled column.
    """
    random_state = check_random_state(random_state)
    if n_samples is None:
        n_samples = X.shape[0]

    min_X, max_X = np.min(X), np.max(X)
    return random_state.uniform(min_X, max_X, size=n_samples)


def generate_synthetic_features(X, method='bootstrap', random_state=1234):
    """generate_synthetic_features

    Generate a synthetic dataset based on the empirical distribution
    of `X`.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        Dataset whose empirical distribution is used to generate the
        synthetic dataset.

    method : str {'bootstrap', 'uniform'}
        Method to use to generate the synthetic dataset. `bootstrap`
        samples each column with replacement. `uniform` generates
        a new column uniformly sampled between the minimum and
        maximum value of each column.

    random_state : int
        Seed to the random number generator.

    Returns
    -------
    synth_X : np.ndarray (n_samples, n_features)
        The synthetic dataset.
    """
    random_state = check_random_state(random_state)
    n_features = int(X.shape[1])

    # XXX: If X is ordinal or nominal then this generates data
    # on a non-discrete scale. Should we sample from a discrete uniform
    # in that case?
    synth_X = np.empty(X.shape, dtype=np.float32)
    for column in range(n_features):
        if method == 'bootstrap':
            synth_X[:, column] = bootstrap_sample_column(
                X[:, column], random_state=random_state)
        elif method == 'uniform':
            synth_X[:, column] = uniform_sample_column(
                X[:, column], random_state=random_state)
        else:
            raise ValueError('method must be either `bootstrap` or `uniform`.')

    return synth_X


def generate_discriminative_dataset(X, method='bootstrap', shuffle=True,
                                    random_state=1234):
    """generate_discriminative_dataset.

    Generate a synthetic dataset based on the empirical distribution
    of `X`. A target column will be returned that is 1 if the row is
    from the real distribution, and 0 if the row is synthetic. The
    number of synthetic rows generated is equal to the number of rows
    in the original dataset.

    Parameters
    ----------
    X : np.ndarray (n_samples, n_features)
        Dataset whose empirical distribution is used to generate the
        synthetic dataset.

    method : str {'bootstrap', 'uniform'}
        Method to use to generate the synthetic dataset. `bootstrap`
        samples each column with replacement. `uniform` generates
        a new column uniformly sampled between the minimum and
        maximum value of each column.

    shuffle : bool, optional (default=True)
        Whether the synthetic dataset should be shuffled. If False then
        the dataset contains original then synthetic samples.

    random_state : int
        Seed to the random number generator.

    Returns
    -------
    X_ : np.ndarray (2 * n_samples, n_features)
        Feature array for the synthetic dataset. The rows
        are randomly shuffled, so synthetic and actual samples should
        be intermixed.

    y_ : np.ndarray (2 * n_samples)
        Target column indicating whether the row is from the actual
        dataset (1) or synthetic (0).
    """
    random_state = check_random_state(random_state)
    n_samples = int(X.shape[0])

    synth_X = generate_synthetic_features(
        X, method=method, random_state=random_state)
    X_ = np.vstack((X, synth_X))
    y_ = np.concatenate((np.ones(n_samples), np.zeros(n_samples)))

    if shuffle:
        permutation_indices = random_state.permutation(np.arange(X_.shape[0]))
        X_ = X_[permutation_indices, :]
        y_ = y_[permutation_indices]

    return X_, y_
