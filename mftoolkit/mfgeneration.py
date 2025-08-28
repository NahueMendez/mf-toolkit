# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 12:22:42 2025

@author: Nahuel Mendez & Sebastian Jaroszewicz
"""
import numpy as np 
import warnings as warning


def generate_fgn(H: float, n_points: int) -> np.ndarray:
    """Generates a time serie of a fractional Gaussian noise (fGn)
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
    [5] Davies, R. B., & Harte, D. S. (1987). Tests for Hurst effect. Biometrika, 74(1), 95-101.
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


def generate_mf_dist (n_points,H:float, power=2.0):
    """Generates a multifractal time series from a probability distribution.

    This function first creates a fractional Gaussian noise (fGn) series
    with long-range correlations defined by the Hurst exponent. Then, it
    applies a non-linear transformation to introduce multifractality.

    Parameters
    ----------
    n_points : int
        The length of the time series to generate.
    H : float
        The Hurst exponent of the underlying correlation structure (0 < H < 1).
    power : float
        The exponent for the non-linear transformation. Values > 1
        introduce multifractality.

    Returns
    -------
    np.ndarray
        The generated 1D multifractal time series.

    Notes
    -----
    The parameter `H` controls the long-range memory (the monofractal
    component), while `power` controls the strength of the multifractality.

    """
    # 1. Generate a monofractal base with linear correlations (fGn)
    fgn_base = generate_fgn(H,n_points)

    # 2. Apply a non-linear transformation to introduce multifractality.
    multifractal_series = np.abs(fgn_base)**power

    return (multifractal_series - np.mean(multifractal_series)) / np.std(multifractal_series)

def generate_mf_corr(n_points: int, a: float = 0.3) -> np.ndarray:
    """Generates a multifractal series with a perfect Gaussian PDF.

    This method disentangles the PDF from the correlation structure. It first
    generates a standard binomial cascade to obtain a multifractal correlation
    structure. It then imposes this structure onto a Gaussian white noise
    series via rank-ordering.

    The final output is a series that retains the intricate, multifractal
    correlation structure of the cascade model while possessing a strictly
    Gaussian PDF.

    Parameters
    ----------
    n_points : int
        The length of the time series to generate. Must be a positive
        integer and a power of 2.
    a : float, optional
        The multiplier for the binomial cascade, by default 0.3. Must be
        in the range (0, 1). Values closer to 0.5 generate weaker
        multifractality.

    Returns
    -------
    np.ndarray
        A 1D NumPy array with the correlation structure of a binomial
        cascade but a Gaussian distribution of values.
    """
    # --- 1. Input Parameter Validation ---
    if not (isinstance(n_points, int) and n_points > 0 and (n_points & (n_points - 1) == 0)):
        # Efficient bitwise check to ensure n_points is a power of 2.
        raise ValueError("n_points must be a positive integer and a power of 2.")

    if not 0 < a < 1:
        raise ValueError("The parameter 'a' must be in the range (0, 1).")

    if a == 0.5:
        # Warn the user if they are generating a monofractal series.
        warnings.warn("a=0.5 will generate a monofractal series.", RuntimeWarning)

    # --- 2. Generate the Canonical Binomial Cascade ---
    # This series will have the desired multifractal correlation structure.
    
    k = int(np.log2(n_points))
    cascade_series = np.ones(n_points)
    b = 1.0 - a

    # Iterate through the k levels of the cascade.
    for i in range(k):
        level_step = n_points // (2**i)
        for j in range(2**i):
            start_index = j * level_step
            mid_index = start_index + level_step // 2
            
            # Apply multiplier 'a' to the first half and 'b' to the second.
            cascade_series[start_index:mid_index] *= a
            cascade_series[mid_index:start_index + level_step] *= b

    # --- 3. Generate Gaussian White Noise ---
    # This series has the desired value distribution (PDF), but no correlations.
    gaussian_series = np.random.randn(n_points)

    # --- 4. Impose the Correlation Structure onto the Gaussian Values ---
    
    # Get the ranks of the cascade series.
    # argsort() returns the indices that would sort the array.
    # Applying it twice gives the rank of each element (from 0 to N-1).
    cascade_ranks = cascade_series.argsort().argsort()
    
    # Sort the values of the Gaussian series.
    sorted_gaussian = np.sort(gaussian_series)
    
    # Reorder the Gaussian series to follow the rank pattern of the cascade.
    # This is the key step that "imprints" the correlation structure.
    final_series = sorted_gaussian[cascade_ranks]

    return final_series