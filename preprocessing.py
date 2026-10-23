import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = np.nanstd(X, axis=0)
        # Avoid division by zero for constant features
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        data_min = np.nanmin(X, axis=0)
        data_max = np.nanmax(X, axis=0)
        data_range = data_max - data_min
        
        self.scale_ = (self.feature_range[1] - self.feature_range[0]) / (data_range + 1e-8)
        self.min_ = self.feature_range[0] - data_min * self.scale_
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X * self.scale_ + self.min_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class SimpleImputer:
    def __init__(self, strategy='mean', fill_value=0.0):
        self.strategy = strategy
        self.fill_value = fill_value
        self.statistics_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.strategy == 'mean':
            self.statistics_ = np.nanmean(X, axis=0)
        elif self.strategy == 'median':
            self.statistics_ = np.nanmedian(X, axis=0)
        elif self.strategy == 'constant':
            self.statistics_ = np.full(X.shape[1], self.fill_value)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Handle cases where an entire column is NaN
        self.statistics_ = np.nan_to_num(self.statistics_, nan=self.fill_value)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64).copy()
        # Vectorized replacement of NaNs
        mask = np.isnan(X)
        # Broadcast statistics to match mask shape
        X[mask] = np.take(self.statistics_, np.where(mask)[1])
        return X

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class OneHotEncoder:
    def __init__(self):
        self.categories_ = None

    def fit(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        self.categories_ = []
        for i in range(X.shape[1]):
            self.categories_.append(np.unique(X[:, i]))
        return self

    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n_samples = X.shape[0]
        encoded_blocks = []
        
        for i in range(X.shape[1]):
            cats = self.categories_[i]
            block = (X[:, i][:, None] == cats[None, :]).astype(float)
            encoded_blocks.append(block)
            
        return np.hstack(encoded_blocks)

    def fit_transform(self, X):
        return self.fit(X).transform(X)