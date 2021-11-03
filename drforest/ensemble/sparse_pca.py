import numpy as np


from sklearn.utils import check_random_state


def normalize(x):
    return  x / np.sqrt((x ** 2).sum())


def random_uniform(n, random_state):
    return normalize(random_state.randn(n, 1))


def power_method(A, max_iter=100, eps=1e-10, random_state=None):
    random_state = check_random_state(random_state)

    x = random_uniform(A.shape[0], random_state=random_state)
    n_features = x.shape[0]
    for _ in range(max_iter):
        x_prev = x.copy()
        x = normalize(A @ x)
        diff = np.abs(x - x_prev).sum() / np.abs(x_prev).sum()
        if np.abs(x - x_prev).sum() < eps:
            break

    return x


def truncated_power_method(A, k, max_iter=100, eps=1e-10, random_state=42):
    n_features = A.shape[0]

    x = power_method(A, max_iter=max_iter, eps=eps, random_state=random_state)
    indices = np.argsort(np.abs(x).ravel())[:n_features-k]
    x[indices, :] = 0
    x = normalize(x)

    for _ in range(max_iter):
        x_prev = x.copy()

        x = normalize(A @ x)
        indices = np.argsort(np.abs(x).ravel())[:n_features-k]
        x[indices, :] = 0
        x = normalize(x)

        diff = np.abs(x - x_prev).sum() / np.abs(x_prev).sum()
        if np.abs(x - x_prev).sum() < eps:
            break

    return x
