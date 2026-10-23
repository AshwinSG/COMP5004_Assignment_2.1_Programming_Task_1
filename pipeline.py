class Transformer:
    def fit(self, X, y=None): return self
    def transform(self, X): raise NotImplementedError
    def fit_transform(self, X, y=None): return self.fit(X, y).transform(X)

class Estimator:
    def fit(self, X, y): raise NotImplementedError
    def predict(self, X): raise NotImplementedError

class Pipeline:
    def __init__(self, steps):
        self.steps = steps

    def fit(self, X, y=None):
        X_transformed = X
        for name, step in self.steps[:-1]:
            X_transformed = step.fit_transform(X_transformed, y)
        
        # Final step
        name, final_step = self.steps[-1]
        if hasattr(final_step, 'fit'):
            final_step.fit(X_transformed, y)
        return self

    def transform(self, X):
        X_transformed = X
        for name, step in self.steps:
            X_transformed = step.transform(X_transformed)
        return X_transformed

    def fit_transform(self, X, y=None):
        X_transformed = X
        for name, step in self.steps:
            X_transformed = step.fit_transform(X_transformed, y)
        return X_transformed

    def predict(self, X):
        X_transformed = self.transform(X)
        return self.steps[-1][1].predict(X_transformed)