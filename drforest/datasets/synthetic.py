import numpy as np

from sklearn.utils import check_random_state


__all__ = ['make_cubic', 'make_quadratic', 'make_simulation1',
           'make_simulation2', 'make_simulation3', 'make_simulation4',
           'make_simulation5']


def make_cubic(n_samples=500, n_features=10, n_informative=2,
               random_state=None):
    """Generates a dataset with a cubic response curve.

    Inputs X are independent normally distributed features. The output y
    is created according to the formula::

        beta = np.hstack((
            np.ones(n_informative), np.zeros(n_features - n_informative)))
        h = np.dot(X, beta)
        y(h) = 0.125 * h ** 3 + 0.5 * N(0, 1).

    Out of the `n_features` features,  only `n_informative` are actually
    used to compute `y`. The remaining features are independent of `y`.
    As such the central subspace is one dimensional and consists of the
    `h` axis.

    Parameters
    ----------
    n_samples : int, optimal (default=500)
        The number of samples.

    n_features : int, optional (default=10)
        The number of features. Should be at least equal to `n_informative`.

    n_informative : int, optional (default=2)
        The number of informative features used to construct h. Should be
        at least 1.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The input samples.

    y : array of shape [n_samples]
        The output values.
    """
    rng = check_random_state(random_state)

    if n_informative < 1:
        raise ValueError("`n_informative` must be >= 1. "
                         "Got n_informative={0}".format(n_informative))

    if n_features < n_informative:
        raise ValueError("`n_features` must be >= `n_informative`. "
                         "Got n_features={0} and n_informative={1}".format(
                            n_features, n_informative))

    # normally distributed features
    X = rng.randn(n_samples, n_features)

    # beta is a linear combination of informative features
    beta = np.hstack((
        np.ones(n_informative), np.zeros(n_features - n_informative)))

    # cubic in subspace
    y = 0.125 * np.dot(X, beta) ** 3
    y += 0.5 * rng.randn(n_samples)

    return X, y


def make_quadratic(n_samples=500, n_features=10, n_informative=2,
                   return_projection=False, random_state=None):
    """Generates a dataset with a quadratic response curve.

    Inputs X are independent normally distributed features. The output y
    is created according to the formula::

        beta = np.hstack((
            np.ones(n_informative), np.zeros(n_features - n_informative)))
        h = np.dot(X, beta)
        y(h) = 0.125 * h ** 2 + 0.5 * N(0, 1).

    Out of the `n_features` features,  only `n_informative` are actually
    used to compute `y`. The remaining features are independent of `y`.
    As such the central subspace is one dimensional and consists of the
    `h` axis.

    Parameters
    ----------
    n_samples : int, optimal (default=500)
        The number of samples.

    n_features : int, optional (default=10)
        The number of features. Should be at least equal to `n_informative`.

    n_informative : int, optional (default=2)
        The number of informative features used to construct h. Should be
        at least 1.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The input samples.

    y : array of shape [n_samples]
        The output values.
    """
    rng = check_random_state(random_state)

    if n_informative < 1:
        raise ValueError("`n_informative` must be >= 1. "
                         "Got n_informative={0}".format(n_informative))

    if n_features < n_informative:
        raise ValueError("`n_features` must be >= `n_informative`. "
                         "Got n_features={0} and n_informative={1}".format(
                            n_features, n_informative))

    # normally distributed features
    X = rng.randn(n_samples, n_features)

    # beta is a linear combination of informative features
    beta = np.hstack((
        np.ones(n_informative), np.zeros(n_features - n_informative)))

    # cubic in subspace
    y = 0.125 * np.dot(X, beta) ** 2
    y += 0.25 * rng.randn(n_samples)

    if return_projection:
        B = beta.reshape(-1, 1)
        P = np.dot(B, np.dot(np.linalg.inv(np.dot(B.T, B)), B.T))
        return X, y, P
    return X, y


def make_simulation1(n_samples=1000, n_features=2,
                     return_projection=False, noise=1,
                     correlate_features=False, random_state=123):
    if n_features < 2:
        raise ValueError("There must be at least 2 features.")

    rng = check_random_state(random_state)


    if correlate_features:
        mean = np.zeros(n_features)
        cov = np.fromfunction(lambda i, j: 0.5 ** np.abs(i - j),
                              (n_features, n_features))
        X = rng.multivariate_normal(mean, cov,
                                    size=n_samples)
    else:
        X = rng.uniform(-3, 3, n_samples * n_features)
        X = X.reshape(n_samples, n_features)

    r1 = X[:, 0] - X[:, 1]
    r2 = X[:, 0] + X[:, 1]

    y = 20 * np.maximum.reduce([np.exp(-2 * r1 ** 2),
                                np.exp(-1 * r2 ** 2),
                                2 * np.exp(-0.5 * (X[:, 0] ** 2 + X[:, 1] ** 2))])

    if noise:
        y += rng.normal(scale=noise, size=y.shape)

    if return_projection:
        beta1 = np.eye(n_features)[0, :]
        beta2 = np.eye(n_features)[1, :]
        B = np.c_[beta1, beta2]
        P = np.dot(B, np.dot(np.linalg.pinv(np.dot(B.T, B)), B.T))
        return X, y, P
    else:
        return X, y


def make_simulation2(n_samples=1000, n_features=10, scale='auto', noise=None,
                 random_state=123):
    rng = check_random_state(random_state)

    X = rng.uniform(-1, 1, n_samples * n_features)
    X = X.reshape(n_samples, n_features)

    r1 = X[:, 0]
    r2 = X[:, 1]
    if scale == 'auto':
        scale = 100. / n_features
    y = 20 * np.maximum.reduce([ np.exp(-0.9 * scale * r1 ** 2),
                                 np.exp(-0.9 * scale * r2 ** 2),
                                1.75 * np.exp(-scale * ((r1 + r2) ** 2)),
                                1.75 * np.exp(-scale * ((r1 - r2) ** 2))
                           ])

    if noise is not None:
        y += rng.normal(scale=noise, size=y.shape)

    return X, y


def make_simulation3(n_samples=1000,
                     noninformative_features=True, correlated_features=True,
                     return_projection=False, random_state=123):
    rng = check_random_state(random_state)
    n_features = 12 if noninformative_features else 6
    if correlated_features:
        mean = np.zeros(n_features)
        cov = np.fromfunction(lambda i, j: 0.5 ** np.abs(i - j),
                              (n_features, n_features))
        X = rng.multivariate_normal(mean, cov,
                                    size=n_samples)
    else:
        X = rng.randn(n_samples, n_features)

    if noninformative_features:
        beta1 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]) / np.sqrt(6)
        beta2 = np.array([1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 0, 0]) / np.sqrt(6)
    else:
        beta1 = np.array([1, 1, 1, 1, 1, 1]) / np.sqrt(6)
        beta2 = np.array([1, -1, 1, -1, 1, -1]) / np.sqrt(6)

    y = np.dot(X, beta1) ** 2 + np.dot(X, beta2) ** 2
    y += 0.5 * rng.normal(size=y.shape)

    if return_projection:
        B = np.c_[beta1, beta2]
        P = np.dot(B, np.dot(np.linalg.pinv(np.dot(B.T, B)), B.T))
        return X, y, P

    return X, y


def make_simulation4(n_samples=1000, correlated_features=True,
                     return_projection=False, random_state=123):
    rng = check_random_state(random_state)

    n_features = 10
    if correlated_features:
        mean = np.zeros(n_features)
        cov = np.fromfunction(lambda i, j: 0.5 ** np.abs(i -j),
                              (n_features, n_features))
        X = rng.multivariate_normal(mean, cov,
                                    size=n_samples)
    else:
        X = rng.randn(n_samples, n_features)

    # construct true betas
    beta1 = np.array([1, 2, 3, 4, 0, 0, 0, 0, 0, 0]) / np.sqrt(30)
    beta2 = np.array([-2, 1, -4, 3, 1, 2, 0, 0, 0, 0]) / np.sqrt(35)
    beta3 = np.array([0, 0, 0, 0, 2, -1, 2, 1, 2, 1]) / np.sqrt(15)
    beta4 = np.array([0, 0, 0, 0, 0, 0, -1, -1, 1, 1]) / 2

    y = np.dot(X, beta1) * np.dot(X, beta2) ** 2
    y += np.dot(X, beta3) * np.dot(X, beta4)
    y += 0.5 * rng.normal(size=y.shape)

    if return_projection:
        B = np.c_[beta1, beta2, beta3, beta4]
        P = np.dot(B, np.dot(np.linalg.pinv(np.dot(B.T, B)), B.T))
        return X, y, P, B

    return X, y


def make_simulation5(n_samples=1000, correlated_features=True,
                     return_projection=False, random_state=123):
    rng = check_random_state(random_state)

    n_features = 2
    if correlated_features:
        mean = np.zeros(n_features)
        cov = np.fromfunction(lambda i, j: 0.5 ** np.abs(i -j),
                              (n_features, n_features))
        X = rng.multivariate_normal(mean, cov,
                                    size=n_samples)
    else:
        X = rng.randn(n_samples, n_features)

    # construct true betas
    beta1 = np.array([1, 3]) / np.sqrt(10)
    beta2 = np.array([-2, 1]) / np.sqrt(5)
    beta3 = np.array([3, -4]) / np.sqrt(25)
    beta4 = np.array([-1, -1]) / np.sqrt(2)

    y = np.dot(X, beta1) * np.dot(X, beta2) ** 2
    y += np.dot(X, beta3) * np.dot(X, beta4)
    y += 0.5 * rng.normal(size=y.shape)

    if return_projection:
        B = np.c_[beta1, beta2, beta3, beta4]
        P = np.dot(B, np.dot(np.linalg.pinv(np.dot(B.T, B)), B.T))
        return X, y, P, B

    return X, y
