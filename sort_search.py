import numpy as np

def stable_sort(arr):
    return np.sort(arr, kind='stable')

def multi_key_sort(arr, keys):
    return arr[np.lexsort(tuple(arr[:, k] for k in reversed(keys)))]

def topk(values, k, largest=True, return_indices=True):
    values = np.asarray(values)
    if k >= len(values):
        if largest:
            idx = np.argsort(values)[::-1]
        else:
            idx = np.argsort(values)
    else:
        if largest:
            idx = np.argpartition(values, -k)[-k:]
            # Sort the top k internally to return descending order
            idx = idx[np.argsort(values[idx])[::-1]]
        else:
            idx = np.argpartition(values, k)[:k]
            idx = idx[np.argsort(values[idx])]
            
    if return_indices:
        return values[idx], idx
    return values[idx]

def quickselect(arr, k):
    arr = arr.copy()
    k = k - 1 # 0-indexed
    
    def partition(left, right, pivot_idx):
        pivot_val = arr[pivot_idx]
        arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]
        store_idx = left
        for i in range(left, right):
            if arr[i] < pivot_val:
                arr[store_idx], arr[i] = arr[i], arr[store_idx]
                store_idx += 1
        arr[right], arr[store_idx] = arr[store_idx], arr[right]
        return store_idx

    left, right = 0, len(arr) - 1
    while left <= right:
        pivot_idx = np.random.randint(left, right + 1)
        pivot_idx = partition(left, right, pivot_idx)
        
        if pivot_idx == k:
            return arr[k]
        elif pivot_idx < k:
            left = pivot_idx + 1
        else:
            right = pivot_idx - 1
    return arr[k]

def binary_search(sorted_array, x):
    idx = np.searchsorted(sorted_array, x, side='left')
    exists = idx < len(sorted_array) and sorted_array[idx] == x
    return idx, exists