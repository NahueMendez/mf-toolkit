# -*- coding: utf-8 -*-
import numpy as np 

def generate_fgn(H: float, n_points: int) -> np.ndarray:
    """
    Generates a time serie of a fractional Gaussian noise (fGn)
    using the Davies-Harte exact method.

    Parameters:
    -----------
    H : float
        Hurst exponent of the serie. Must be in the range (0, 1).
    n_points : int
        Length (number of points) of the time series to generate.

    Returns:
    --------
    fgn_series : np.ndarray
        A 1D NumPy array containing the fGn time serie.

    References:
    -----------
    [4] Davies, R. B., & Harte, D. S. (1987). Tests for Hurst effect. Biometrika, 74(1), 95-101.
    """
    if not 0 < H < 1:
        raise ValueError("Hurst exponent must be in range (0, 1).")
    if not isinstance(n_points, int) or n_points <= 0:
        raise ValueError("n_points must be a positive integer number")

    # El método de Davies-Harte requiere un grid extendido de tamaño 2*(N-1)
    # para la incrustación circulante.
    N = n_points - 1
    M = 2 * N

    # Calculate ACVF
    k = np.arange(0, N + 1)
    gamma_k = 0.5 * (np.abs(k - 1)**(2*H) - 2 * np.abs(k)**(2*H) + np.abs(k + 1)**(2*H))

    # First row of circulant matrix
    circulant_row = np.concatenate([gamma_k, gamma_k[N-1:0:-1]])

    # Eigen values calculation via FFT for circulant_row
    lambda_val = np.fft.fft(circulant_row).real
    if np.any(lambda_val < 0):
        lambda_val[lambda_val < 0] = 0

    # Generate complex noise in frequency domain
    W = np.random.normal(0, 1, M) + 1j * np.random.normal(0, 1, M)

    # Adjust power spectrum
    f_k = np.fft.fft(W * np.sqrt(lambda_val / (2 * M)))

    # Take the real-part.
    fgn_series = f_k.real[:N+1]
    
    return fgn_series
