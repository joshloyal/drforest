from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from sklearn.utils import check_random_state


__all__ = ['make_cubic', 'make_quadratic', 'make_mexican_hat',
           'make_mexican_hat2', 'make_ellipse', 'make_ellipses',
           'make_mave_2', 'make_dmave_1', 'make_text_1',
           'make_ma_models', 'make_lass', 'linear_sine',
           'make_crosses', 'make_strong_linear', 'make_camel']


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


def make_mexican_hat(n_samples=500, random_state=None, noise=0.0):
    """The Mexican Hat Function

    y = cos(5 sqrt(X[:, 0]**2 + X[:, 1]**2)) * exp(-(X[:, 0]**2 + X[:, 1]**2))

    """
    rng = check_random_state(random_state)

    # normally distributed features
    X = rng.randn(n_samples, 5)

    r = X[:, 0]**2 + X[:, 1]**2
    y = np.cos(5 * np.sqrt(r)) * np.exp(-r) + noise

    if noise > 0.0:
        y += rng.normal(scale=noise, size=y.shape)

    return X, y


def make_mexican_hat2(n_samples=500, random_state=None, noise=0.0):
    """The Mexican Hat Function

    Z_1 = X[:, 0] + X[:, 1]
    Z_2 = X[:, 0] + X[:, 2]

    y = cos(5 sqrt(Z_1**2 + Z_2**2)) * exp(-(Z_1[:, 0]**2 + Z_2**2))

    """
    rng = check_random_state(random_state)

    # normally distributed features
    X = rng.randn(n_samples, 30)

    Z_1 = X[:, 0] + X[:, 1]
    Z_2 = X[:, 2] + X[:, 3]
    r = Z_1 ** 2 + Z_2 ** 2
    y = np.cos(5 * np.sqrt(r)) * np.exp(-r)

    if noise > 0.0:
        y += rng.normal(scale=noise, size=y.shape)

    return X, y


def make_ellipse(n_samples=1000, n_features=4, a1=1.0, a2=1.0, noise=0.0,
                 return_projection=False, random_state=123):
    rng = check_random_state(random_state)

    if n_features < 2:
        raise ValueError("`n_features` must be >= 2. "
                         "Got n_features={0}".format(n_features))

    # sample X uniformly in the box [-1, 1] x [-1, 1]
    X = rng.uniform(
        -1, 1, n_samples * n_features).reshape(n_samples, n_features)

    # axes of the ellipse
    r1 = X[:, 0] - X[:, 1]
    r2 = X[:, 0] + X[:, 1]

    y = a1 * r1 ** 2 + a2 * r2 ** 2

    if noise > 0.0:
        y += rng.normal(scale=noise, size=y.shape)

    if return_projection:
        B = np.zeros((n_features, 2))
        B[0, 0] = 1 / np.sqrt(2)
        B[0, 1] = 1 / np.sqrt(2)
        B[1, 0] = -1 / np.sqrt(2)
        B[1, 1] = 1 / np.sqrt(2)
        P = np.dot(B, np.dot(np.linalg.pinv(np.dot(B.T, B)), B.T))
        return X, y, P
    return X, y


def make_ma_models(model=4, n_samples=1000,
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

    if model == 1:
        y = np.dot(X, beta1) / (0.5 + (np.dot(X, beta2) + 1.5) ** 2)
        y += 0.5 * rng.normal(scale=0.5, size=y.shape)
    elif model == 2:
        y = np.dot(X, beta1) ** 2 + 2 * np.abs(np.dot(X, beta2))
        y += 0.1 * np.abs(np.dot(X, beta2)) * rng.normal(size=y.shape)
    elif model == 3:
        y = np.exp(np.dot(X, beta1)) + 2 * (np.dot(X, beta2) + 1) ** 2
        y += np.abs(np.dot(X, beta1)) * rng.normal(size=y.shape)
    else:
        y = np.dot(X, beta1) ** 2 + np.dot(X, beta2) ** 2
        y += 0.5 * rng.normal(size=y.shape)

    if return_projection:
        B = np.c_[beta1, beta2]
        P = np.dot(B, np.dot(np.linalg.pinv(np.dot(B.T, B)), B.T))
        return X, y, P

    return X, y


def make_ellipses(n_samples=1000, n_features=4, a1=1.0, a2=1.0, noise=0.0,
                  random_state=123):
    rng = check_random_state(random_state)

    if n_features < 4:
        raise ValueError("`n_features` must be >= 4. "
                         "Got n_features={0}".format(n_features))

    # sample X uniformly in the box [0, 1] x [0, 1]
    X = rng.uniform(
        0, 1, n_samples * n_features).reshape(n_samples, n_features)

    # axes of the top level subspace
    U1 = (X[:, 0] - 0.25) - (X[:, 1] - 0.75)
    U2 = (X[:, 0] - 0.25) + (X[:, 1] - 0.75)

    y = (1/np.sqrt(0.1)) * np.exp(((-5.0 * U1**2 - U2**2)) / np.sqrt(0.1))

    U1 = (X[:, 1] - 0.5)
    U2 = (X[:, 0] - 0.75)
    y += (1/np.sqrt(0.1)) * np.exp(((-U1**2 - 15.0 * U2**2)) / np.sqrt(0.1))

    if noise > 0.0:
        y += rng.normal(scale=noise, size=y.shape)

    return X, y


def make_mave_2(n_samples=1000, correlated_features=True,
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

def make_dmave_1(n_samples=1000, n_features=30, correlated_features=False,
                 return_projection=False, random_state=123):
    rng = check_random_state(random_state)

    if correlated_features:
        mean = np.zeros(n_features)
        cov = np.fromfunction(lambda i, j: 0.5 ** np.abs(i -j), (n_features, n_features))
        X = rng.multivariate_normal(mean, cov,
                                    size=n_samples)
    else:
        X = rng.randn(n_samples, n_features)

    # construct true betas
    #beta1 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]) / 2
    beta1 = np.zeros(n_features)
    beta1[:4] = 0.5
    beta2 = np.zeros(n_features)
    beta2[:4] = np.array([1, -1, 1, -1]) / 2
    #beta2 = np.array([1, -1, 1, -1, 0, 0, 0, 0, 0, 0]) / 2

    y = np.sign(2 * np.dot(X, beta1) + rng.normal(scale=1, size=n_samples))
    y *= np.log(np.abs(2 * np.dot(X, beta2) + 4 + rng.normal(scale=1, size=n_samples)))

    if return_projection:
        B = np.c_[beta1, beta2]
        P = np.dot(B, np.dot(np.linalg.pinv(np.dot(B.T, B)), B.T))
        return X, y, P
    return X, y

def make_text_1(n_samples=1000, n_features=30, correlated_features=False,
                return_projection=False, random_state=123):
    rng = check_random_state(random_state)

    if correlated_features:
        mean = np.zeros(n_features)
        cov = np.fromfunction(lambda i, j: 0.5 ** np.abs(i -j), (n_features, n_features))
        X = rng.multivariate_normal(mean, cov,
                                    size=n_samples)
    else:
        X = rng.randn(n_samples, n_features)

    # construct true betas
    #beta1 = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0]) / 2
    beta1 = np.zeros(n_features)
    beta1[:4] = 0.5
    beta2 = np.zeros(n_features)
    beta2[:4] = np.array([1, -1, 1, -1]) / 2
    #beta2 = np.array([1, -1, 1, -1, 0, 0, 0, 0, 0, 0]) / 2

    y = 2 * np.sin(1.4 * np.dot(X, beta1))
    y += (np.dot(X, beta2) + 1) ** 2
    y += rng.normal(scale=0.4, size=y.shape)

    if return_projection:
        B = np.c_[beta1, beta2]
        P = np.dot(B, np.dot(np.linalg.pinv(np.dot(B.T, B)), B.T))
        return X, y, P
    return X, y


def make_single_index(n_samples=1000, n_features=30, correlated_features=False,
                      return_projection=False, random_state=123):
    rng = check_random_state(random_state)

    if correlated_features:
        mean = np.zeros(n_features)
        cov = np.fromfunction(lambda i, j: 0.5 ** np.abs(i -j), (n_features, n_features))
        X = rng.multivariate_normal(mean, cov,
                                    size=n_samples)
    else:
        X = rng.randn(n_samples, n_features)

    rng = check_random_state(random_state)

    if correlated_features:
        mean = np.zeros(n_features)
        cov = np.fromfunction(lambda i, j: 0.5 ** np.abs(i -j), (n_features, n_features))
        X = rng.multivariate_normal(mean, cov,
                                    size=n_samples)
    else:
        X = rng.randn(n_samples, n_features)

    beta = np.zeros(n_features)


def make_lass(n_samples=1000, n_features=2,
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


def linear_sine(n_samples=1000, n_features=10, noise=1.0, random_state=123):
    rng = check_random_state(random_state)

    X = rng.randn(n_samples, n_features)
    y = X[:, 0] + X[:, 5] + X[:, 6]
    y += np.sin(X[:, 3] + 2 * X[:, 5])
    y += (X[:, 1] - X[:, 2] + X[:, 3] - X[:, 4]) ** 2
    y += rng.normal(scale=noise, size=y.shape)

    return X, y


def make_crosses(n_samples=1000, n_features=10, scale='auto', noise=None,
                 random_state=123):
    rng = check_random_state(random_state)

    X = rng.uniform(-1, 1, n_samples * n_features)
    #X = rng.randn(n_samples, n_features)
    X = X.reshape(n_samples, n_features)

    #y = 20 * (np.sum(X[:, :5], axis=1) - 0.5) ** 3 + 10 * np.sum(X[:, 5:], axis=1)

    #r1 = np.sum(X[:, :int(n_features / 2)], axis=1)
    #r2 = np.sum(X[:, int(n_features/2):], axis=1)
    #r1 = np.sum(X[:, :3], axis=1)
    #r2 = np.sum(X[:, 3:5], axis=1)
    r1 = X[:, 0]
    r2 = X[:, 1]
    #y = np.maximum.reduce([np.exp(-5 * r1 ** 2),
    #                       np.exp(-20 * r2 ** 2),
    #                       1.75 * np.exp(-3 * ((r1 + r2) ** 2))])
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


def make_strong_linear(n_samples=1000, n_features=10, noise=None, random_state=123):
    rng = check_random_state(random_state)

    X = rng.uniform(0, 1, n_samples * n_features)
    X = X.reshape(n_samples, n_features)

    y = 20 * (X[:, 0] - 0.5) ** 3 + 10 * (X[:, 1] - 0.5) ** 2
    y += 10 * np.sum(X[:, 2:5], axis=1)
    y += 5 * np.sum(X[:, 5:], axis=1)

    if noise is not None:
        y += rng.normal(scale=noise, size=y.shape)

    return X, y


def make_camel(n_samples=1000, n_features=2, noise=None, random_state=42):
    rng = check_random_state(random_state)

    X = rng.uniform(-1, 1, n_samples * n_features)
    X = X.reshape(n_samples, n_features)
    X[:, 0] *= 1.15
    X[:, 1] *= 1.75

    y = (4 - 2.1 * X[:, 1] ** 2 + (1/3.) * X[:, 1] ** 4) * X[:, 1] ** 2
    y += X[:, 0] * X[:, 1]
    y += (-4 + 4 * X[:, 0] ** 2) * X[:, 0] ** 2

    if noise is not None:
        y += rng.normal(scale=noise, size=y.shape)

    return X, y
