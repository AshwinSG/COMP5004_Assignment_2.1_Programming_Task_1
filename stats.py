import numpy as np

def mean(data, axis=None):
    return np.nanmean(data, axis=axis)

def median(data, axis=None):
    return np.nanmedian(data, axis=axis)

def std(data, axis=None, ddof=0):
    return np.nanstd(data, axis=axis, ddof=ddof)

def quantiles(data, q, axis=None):
    return np.nanquantile(data, q, axis=axis)

def histogram(data, bins=10, range=None):
    return np.histogram(data[~np.isnan(data)], bins=bins, range=range)

class WelfordStats:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, new_values):
        new_values = np.asarray(new_values)
        for x in new_values:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.M2 += delta * delta2

    def update_vectorized(self, new_values):
        new_values = np.asarray(new_values)
        n_b = len(new_values)
        if n_b == 0: return
        
        mean_b = np.mean(new_values)
        M2_b = np.sum((new_values - mean_b)**2)
        
        if self.count == 0:
            self.count = n_b
            self.mean = mean_b
            self.M2 = M2_b
        else:
            delta = mean_b - self.mean
            n_total = self.count + n_b
            self.mean += delta * n_b / n_total
            self.M2 += M2_b + delta**2 * self.count * n_b / n_total
            self.count = n_total

    @property
    def variance(self):
        if self.count < 2: return 0.0
        return self.M2 / (self.count - 1) # Sample variance

    @property
    def std(self):
        return np.sqrt(self.variance)