import numpy as np

def ackley(x, a=20, b=0.2, c=2 * np.pi):
    # Ensure x is a numpy array for easy vectorized operations
    x = np.array(x)
    d = x.size  # Dimension of the input

    # Compute the first term
    sum_sq_term = -a * np.exp(-b * np.sqrt(np.sum(x**2) / d))
    
    # Compute the second term
    cos_term = -np.exp(np.sum(np.cos(c * x)) / d)
    
    # Ackley function result
    result = sum_sq_term + cos_term + a + np.exp(1)
    return result

def levy(x):
    # Ensure x is a numpy array
    x = np.array(x)
    d = x.size  # Dimension of the input

    # Compute w values for each x element
    w = 1 + (x - 1) / 4

    # Compute the first term
    term1 = np.sin(np.pi * w[0])**2

    # Compute the middle sum term
    term_sum = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))

    # Compute the last term
    term_last = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)

    # Levy function result
    result = term1 + term_sum + term_last
    return result

def ideal_square_wave(t):
    """Ideal square wave function with period 2π."""
    return np.where(np.sin(t) >= 0, 1, -1)

def square_wave_fourier_series(t, p, n=10):
    """
    Approximate square wave using Fourier series up to n_terms.
    n_terms specifies the number of terms (odd harmonics) in the series.
    """
    result = p[0]
    for i in range(1, n):
        result += p[i]*np.cos(i*t) + p[i+n]*np.sin(i*t)
    return result

def cal_score(T, p, n=10):
    n_sample = len(T)
    result = 0
    for t in T:
        result += np.abs(square_wave_fourier_series(t, p, n) - ideal_square_wave(t))/n_sample
    return result