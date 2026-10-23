import numpy as np

def rank(data, method='average'):
    data = np.asarray(data)
    sorter = np.argsort(data)
    inv_sorter = np.empty_like(sorter)
    inv_sorter[sorter] = np.arange(len(data))
    
    sorted_data = data[sorter]
    
    unique_vals, unique_counts = np.unique(sorted_data, return_counts=True)
    
    if method == 'ordinal':
        return inv_sorter.astype(float) + 1
    
    elif method == 'dense':
        # Map unique values to dense ranks
        dense_ranks = np.arange(1, len(unique_vals) + 1)
        # Create a mapping array
        max_val = int(unique_vals.max()) if unique_vals.size > 0 else 0
        ranks = dense_ranks[np.searchsorted(unique_vals, data)]
        return ranks
        
    elif method == 'average':
        # Calculate average rank for ties
        # Cumulative sum of counts gives the upper bound of ranks for each unique value
        cum_counts = np.cumsum(unique_counts)
        # The average rank for a group of size c starting after 'prev_cum' is (prev_cum + 1 + cum_counts) / 2
        prev_cum = cum_counts - unique_counts
        avg_ranks = (prev_cum + 1 + cum_counts) / 2.0
        return avg_ranks[np.searchsorted(unique_vals, data)]
        
    else:
        raise ValueError(f"Unknown rank method: {method}")

def percentile(data, q, interpolation='linear'):
    data = np.asarray(data)
    q = np.asarray(q)
    
    # Sort data
    sorted_data = np.sort(data, axis=0)
    n = len(sorted_data)
    
    # Calculate rank index
    rank_idx = (q / 100.0) * (n - 1)
    lower_idx = np.floor(rank_idx).astype(int)
    upper_idx = np.ceil(rank_idx).astype(int)
    
    if interpolation == 'lower':
        return sorted_data[lower_idx]
    elif interpolation == 'higher':
        return sorted_data[upper_idx]
    elif interpolation == 'midpoint':
        return (sorted_data[lower_idx] + sorted_data[upper_idx]) / 2.0
    elif interpolation == 'linear':
        # Linear interpolation
        weight = rank_idx - lower_idx
        return sorted_data[lower_idx] * (1 - weight) + sorted_data[upper_idx] * weight
    else:
        raise ValueError(f"Unknown interpolation: {interpolation}")