# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 10:14:32 2025

@author: Nahuel Mendez & Sebastian Jaroszewicz
"""

import numpy as np
import multiprocessing
import warnings, logging
logger = logging.getLogger(__name__)
 

def shuffle_surrogate(original_series: np.ndarray, num_shuffles: int = 100) -> list[np.ndarray]:
    """ Generate surrogate time series by randomly shuffling the original series.

    This method creates surrogate series that have the exact same amplitude
    distribution (histogram) as the original series. However, it destroys all
    temporal structures, including both linear and non-linear correlations,
    by randomly reordering the data points.

    Parameters
    ----------
    original_series : array_like
        The 1D input time series to create surrogates from.
    num_shuffles : int, optional
        The number of shuffled surrogate series to generate.
        Defaults to 100.

    Returns
    -------
    np.ndarray
        A single 1D array representing the average of all surrogates.

    """
    # Ensure the input is a NumPy array for consistent handling
    series_data = np.asarray(original_series)
    
    # Use a list comprehension for a concise and efficient loop
    shuffle = [np.random.permutation(series_data) for _ in range(num_shuffles)]
    #.Calculate the average of all shuffled series
    average_shuffle = np.mean(shuffle, axis=0)
    return average_shuffle




def _iaaft_worker(args):
    """
    Worker function. Generate a surrogate time series using the IAAFT algorithm.

    This method creates a surrogate series that has the same power spectrum
    (and thus the same linear autocorrelation) and the same amplitude
    distribution (histogram) as the original series. It is used to create
    a null model for hypothesis testing, where any nonlinear structure
    present in the original data is destroyed.
    """
    original_series, max_iter, tol, seed = args
    
    # Using a seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    n = len(original_series)
    original_fft = np.fft.rfft(original_series)
    original_amplitudes = np.abs(original_fft)
    sorted_original = np.sort(original_series)
    
    surrogate = np.random.permutation(original_series)

    prev_spec_err = np.inf
    for i in range(max_iter):
        surrogate_fft = np.fft.rfft(surrogate)
        surrogate_phases = np.angle(surrogate_fft)
        new_fft = original_amplitudes * np.exp(1j * surrogate_phases)
        candidate = np.fft.irfft(new_fft, n=n)
        
        ranks = candidate.argsort().argsort()
        surrogate = sorted_original[ranks]

        current_fft = np.fft.rfft(surrogate)
        spec_err = np.mean((original_amplitudes - np.abs(current_fft))**2)

        if prev_spec_err > 0 and abs(prev_spec_err - spec_err) / prev_spec_err < tol:
            break
        prev_spec_err = spec_err
        
    if i == max_iter - 1:
        # We use warnings
        warnings.warn(f"IAAFT surrogate for seed {seed} reached max_iter ({max_iter}) without explicit convergence.", RuntimeWarning)

    return surrogate



def generate_iaaft(original_series, num_surrogates=1, max_iter=1000, tol=1e-8, n_jobs=-1):
    """Generates one or more surrogate time series using the IAAFT algorithm.

    This method creates surrogate series that preserve the power spectrum
    (autocorrelation) and the amplitude distribution of the original series.
    The generation can be run in parallel on multiple CPU cores.

    Parameters
    ----------
    original_series : np.ndarray
        The 1D input time series to create surrogates from.
    num_surrogates : int, optional
        The number of surrogate series to generate. Defaults to 1.
    max_iter : int, optional
        Maximum number of iterations for the algorithm per surrogate.
        Defaults to 1000.
    tol : float, optional
        Tolerance for convergence. The iteration for a surrogate stops if
        the relative change in spectrum error is less than this value.
        Defaults to 1e-8.
    n_jobs : int, optional
        The number of CPU cores to use for parallel generation. -1 means
        using all available cores. If set to 1, runs on a single thread.
        Defaults to -1.

    Returns
    -------
    np.ndarray or list[np.ndarray]
        - If num_surrogates is 1 (default), returns a single NumPy array.
        - If num_surrogates > 1, returns a list of NumPy arrays.

    Notes
    -----
    The Iterative Amplitude Adjusted Fourier Transform (IAAFT) algorithm
    iteratively adjusts the surrogate's amplitudes and power spectrum to
    match the original series [3]. It is used to test against the null hypothesis
    of a stationary linear stochastic process with a possibly non-Gaussian
    distribution of values. This implementation uses Python's `multiprocessing`
    module to parallelize the generation of multiple surrogates. 

    References
    ----------
    [3] Schreiber, T., & Schmitz, A. (2000). Surrogate time series.
        Physica D: Nonlinear Phenomena, 142(3-4), 346-382."""
    
    original_series = np.asarray(original_series)
    if original_series.ndim != 1:
        raise ValueError("Input 'original_series' must be a one-dimensional array.")

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    tasks = [(original_series, max_iter, tol, i) for i in range(num_surrogates)]

    if num_surrogates == 1 or n_jobs == 1:
        logger.info(f"Generating {num_surrogates} surrogate(s) on a single core...")
        surrogate_list = [_iaaft_worker(task) for task in tasks]
    else:
        logger.info(f"Generating {num_surrogates} surrogate(s) in parallel using {n_jobs} cores...")
        with multiprocessing.Pool(processes=n_jobs) as pool:
            surrogate_list = pool.map(_iaaft_worker, tasks)
    
    return surrogate_list[0] if num_surrogates == 1 else surrogate_list


