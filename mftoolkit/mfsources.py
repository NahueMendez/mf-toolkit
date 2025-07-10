# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 10:14:32 2025

@author: Nahuel Mendez & Sebastian Jaroszewicz
"""

import numpy as np
import warnings, logging
logger = logging.getLogger(__name__)
 
def iaaft_surrogate(original_series: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    
    """Generate a surrogate time series using the IAAFT algorithm.

    This method creates a surrogate series that has the same power spectrum
    (and thus the same linear autocorrelation) and the same amplitude
    distribution (histogram) as the original series. It is used to create
    a null model for hypothesis testing, where any nonlinear structure
    present in the original data is destroyed.

    Parameters
    ----------
    original_series : array_like
        The 1D input time series to create a surrogate from.
    max_iter : int, optional
        Maximum number of iterations for the IAAFT algorithm.
        Defaults to 1000.
    tol : float, optional
        Tolerance for convergence. The iteration stops if the relative
        change in the spectrum error is less than this value.
        Defaults to 1e-8.

    Returns
    -------
    np.ndarray
        The generated surrogate time series.

    Notes
    -----
    The Iterative Amplitude Adjusted Fourier Transform (IAAFT) algorithm is
    an improvement over the simple phase-randomized surrogate method.
    It iteratively adjusts the surrogate's amplitudes to match the original
    distribution and adjusts the surrogate's power spectrum to match the
    original spectrum, providing a more accurate surrogate for hypothesis
    testing against a linear stochastic process with possible non-Gaussian
    distribution of values [3].

    References
    ----------
    [3] Schreiber, T., & Schmitz, A. (2000). Surrogate time series.
        Physica D: Nonlinear Phenomena, 142(3-4), 346-382. 
        doi:10.1016/s0167-2789(00)00043-9 
    """
    
    # 1. Obtener las propiedades de la serie original
    original_series = np.asarray(original_series)
    n = len(original_series)

    # Amplitudes de Fourier de la serie original (esto es lo que queremos preservar)
    original_fft = np.fft.rfft(original_series)
    original_amplitudes = np.abs(original_fft)

    # Distribución de valores de la serie original (esto también queremos preservarlo)
    sorted_original = np.sort(original_series)

    # 2. Inicialización
    # Comenzamos con una permutación aleatoria de la serie original.
    # Esto ya tiene la distribución de valores correcta, pero el espectro incorrecto.
    surrogate = np.random.permutation(original_series)

    # 3. Bucle iterativo
    prev_spec_err = np.inf
    for i in range(max_iter):
        # 3a. Imponer el espectro de potencias de la serie original
        # Tomamos la FFT del surrogate actual
        surrogate_fft = np.fft.rfft(surrogate)

        # Obtenemos las fases del surrogate
        surrogate_phases = np.angle(surrogate_fft)

        # Creamos una nueva FFT combinando las amplitudes ORIGINALES con las fases del SURROGATE
        new_fft = original_amplitudes * np.exp(1j * surrogate_phases)

        # Invertimos la FFT para obtener una nueva serie candidata en el dominio del tiempo
        candidate = np.fft.irfft(new_fft, n=n)

        # 3b. Imponer la distribución de valores de la serie original
        # Ordenamos la serie candidata y la serie original.
        # Luego, reemplazamos cada valor de la candidata por el valor de la original
        # que tiene el mismo rango (rank).
        ranks = candidate.argsort().argsort()
        surrogate = sorted_original[ranks]

        # 4. Chequeo de convergencia
        current_fft = np.fft.rfft(surrogate)
        current_amplitudes = np.abs(current_fft)
        spec_err = np.mean((original_amplitudes - current_amplitudes)**2)

        if prev_spec_err > 0 and np.abs(prev_spec_err - spec_err) / prev_spec_err < tol:
            logger.info(f"Convergence reached at iteration: {i+1}.")
            break
        prev_spec_err = spec_err

    if i == max_iter - 1:
        warnings.warn(f"The maximum number of iterations ({max_iter}) without explicit convergence was reached.")

    return surrogate



def shuffle_surrogate(original_series: np.ndarray, num_shuffles: int = 100) -> list[np.ndarray]:
    """Generate surrogate time series by randomly shuffling the original series.

    This method creates surrogate series that have the exact same amplitude
    distribution (histogram) as the original series. However, it destroys all
    temporal structures, including both linear and non-linear correlations,
    by randomly reordering the data points.

    Parameters
    ----------
    original_series : array_like
        The 1D input time series to create surrogates from.
    num_surrogates : int, optional
        The number of shuffled surrogate series to generate.
        Defaults to 100.

    Returns
    -------
    list of np.ndarray
        A list containing the generated surrogate time series. Each element
        of the list is a NumPy array.

    Notes
    -----
    Shuffled surrogates are used to test the null hypothesis (H0) that the
    observed data is indistinguishable from an Independent and Identically
    Distributed (IID) random process. If a metric calculated on the
    original series falls outside the distribution of the same metric
    calculated on these surrogates, it suggests the presence of some form of
    temporal structure or memory in the data.
    This is a less constrained null hypothesis than that of IAAFT surrogates,
    which preserve the linear correlation (power spectrum).
    """
    # Ensure the input is a NumPy array for consistent handling
    series_data = np.asarray(original_series)
    
    # Use a list comprehension for a concise and efficient loop
    shuffle = [np.random.permutation(series_data) for _ in range(num_shuffles)]
    
    return shuffle


import numpy as np
from joblib import Parallel, delayed
import os

def _generate_single_surrogate(original_data, detrend=False, seed=None):
    """
    Generate a surrogate time series using the IAAFT algorithm.

    This method creates a surrogate series that has the same power spectrum
    (and thus the same linear autocorrelation) and the same amplitude
    distribution (histogram) as the original series. It is used to create
    a null model for hypothesis testing, where any nonlinear structure
    present in the original data is destroyed.
    """
    # Seed the random number generator for reproducibility in parallel processes
    if seed is not None:
        np.random.seed(seed)

    L = original_data.size

    # --- Detrend, if requested ---
    if detrend:
        x0 = original_data[0]
        xN = original_data[L - 1]
        time_idx = np.arange(L)
        # Create a copy to avoid modifying the array passed between processes
        data_to_process = np.copy(original_data) - time_idx * (xN - x0) / (L - 1.0)
    else:
        data_to_process = original_data

    # --- Assess spectrum & distribution of original sequence ---
    spectrum_magnitude_original = np.abs(np.fft.rfft(data_to_process))
    distribution_original = np.sort(data_to_process)

    # --- Starting conditions ---
    surrogate_timeseries = np.random.permutation(data_to_process)
    
    # --- Iterative algorithm ---
    # Using a simpler convergence check for internal loop
    for _ in range(100): # Max 100 iterations per surrogate
        surrogate_fft = np.fft.rfft(surrogate_timeseries)
        phases_surrogate = np.angle(surrogate_fft)
        
        # Combine original amplitudes with surrogate phases
        spectrum_surrogate = spectrum_magnitude_original * np.exp(1j * phases_surrogate)
        surrogate_timeseries_freq_adj = np.fft.irfft(spectrum_surrogate, n=L)

        # Impose original amplitude distribution
        ranks = surrogate_timeseries_freq_adj.argsort().argsort()
        surrogate_timeseries = distribution_original[ranks]
        
    return surrogate_timeseries


def generate_iaaft_surrogate(original_data, num_surrogates=50, detrend=False, verbose=True, n_jobs=-1):
    """
    Generates multiple surrogate time series using the IAAFT algorithm in parallel
    and returns the average of all generated surrogates.

    Parameters
    ----------
    original_data : array_like
        The 1D input time series to create surrogates from.
    num_surrogates : int, optional
        Number of surrogate time series to generate before averaging. 
        Default is 50.
    detrend : bool, optional
        Specifies whether the time series has to be detrended prior to
        surrogate generation. Default is False.
    verbose : bool, optional
        Sets the verbosity of the function. If True (default), progress
        messages are displayed.
    n_jobs : int, optional
        The number of CPU cores to use for parallel generation. -1 means
        using all available cores. Default is -1.

    Returns
    -------
    np.ndarray
        A single 1D array representing the average of all generated surrogates.

    Notes
    -----
    This function uses the 'joblib' library for parallel processing to speed up
    the generation of multiple independent surrogates. Each surrogate preserves
    the power spectrum and amplitude distribution of the original series.

    References
    ----------
    [3] Schreiber, T., & Schmitz, A. (1996). Improved surrogate data for
        nonlinearity tests. Physical review letters, 77(4), 635.
    """
    # --- Input validation ---
    if not isinstance(original_data, np.ndarray):
        original_data = np.asarray(original_data)
    if original_data.ndim != 1:
        raise ValueError('Function argument "original_data" must be a one-dimensional numpy array.')
    
    # --- Information and Parallel Execution ---
    if verbose:
        num_cores = n_jobs if n_jobs != -1 else os.cpu_count()
        print(f"Starting IAAFT routine to generate {num_surrogates} surrogates on {num_cores} cores\n")

    # Use joblib.Parallel to run the surrogate generation in parallel
    # The 'verbose' parameter provides a progress bar
    surrogate_list = Parallel(n_jobs=n_jobs, verbose=1 if verbose else 0)(
        delayed(_generate_single_surrogate)(original_data, detrend, seed=i)
        for i in range(num_surrogates)
    )

    # --- Averaging and Returning ---
    if not surrogate_list:
        print("Error: Parallel generation returned no surrogates.")
        return np.array([])
        
    # Convert list of arrays to a 2D array and compute the mean along axis 0
    surrogates_array = np.array(surrogate_list)
    averaged_surrogate = np.mean(surrogates_array, axis=0)

    if verbose:
        print(f"Successfully generated and averaged {len(surrogate_list)} surrogates.\n")

    return averaged_surrogate


