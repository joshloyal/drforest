from sklearn.base import BaseEstimator, TransformerMixin


from drforest.ensemble import DimensionReductionForestRegressor


class DimensionReductionForestKernel(BaseEstimator, TransformerMixin):
    """This is a wrapper class designed to transfer the learned
    kernel from a dimension reduction forest for us in an sklearn pipeline.
    """
    def __init__(self,
                 n_estimators=100,
                 n_slices=10,
                 max_depth=None,
                 max_features="auto",
                 max_directions="auto",
                 min_samples_leaf=2,
                 min_samples_save=50,
                 oob_error=False,
                 estimate_sdr=False,
                 compute_kernel=False,
                 random_state=123,
                 n_jobs=1):
        self.n_estimators = n_estimators
        self.n_slices = n_slices
        self.max_features = max_features
        self.max_directions = max_directions
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_save = min_samples_save
        self.random_state = random_state
        self.oob_error = oob_error
        self.estimate_sdr = estimate_sdr
        self.compute_kernel = compute_kernel
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.forest_ = DimensionReductionForestRegressor(
            n_estimators=self.n_estimators,
            n_slices=self.n_slices,
            max_features=self.max_features,
            max_directions=self.max_directions,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            min_samples_save=self.min_samples_save,
            random_state=self.random_state,
            oob_error=self.oob_error,
            compute_kernel=self.compute_kernel,
            n_jobs=self.n_jobs).fit(X, y)

        return self

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.forest_(X)

    def transform(self, X):
        return self.forest_(X, self.forest_.X_fit_)

