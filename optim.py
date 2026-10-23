import numpy as np

def grad(f, x, h=1e-5, method='central'):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    grad_vec = np.zeros(n)
    
    if method == 'central':
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            grad_vec[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    elif method == 'forward':
        f0 = f(x)
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += h
            grad_vec[i] = (f(x_plus) - f0) / h
    else:
        raise ValueError("Method must be 'central' or 'forward'")
        
    return grad_vec

def jacobian(F, x, h=1e-5, method='central'):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    F0 = np.asarray(F(x))
    m = len(F0)
    J = np.zeros((m, n))
    
    if method == 'central':
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            J[:, i] = (F(x_plus) - F(x_minus)) / (2 * h)
    elif method == 'forward':
        for i in range(n):
            x_plus = x.copy()
            x_plus[i] += h
            J[:, i] = (F(x_plus) - F0) / h
            
    return J