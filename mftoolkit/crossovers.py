# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 12:49:42 2025

@author: Nahuel Mendez & Sebastian Jaroszewicz
"""

# -*- coding: utf-8 -*-
# crossover_analysis.py

import numpy as np
from sklearn.linear_model import LinearRegression
import itertools
import multiprocessing
import os
from scipy.signal import find_peaks
import warnings,logging
logger = logging.getLogger(__name__)

# --- 1. SAFE NUMBA IMPORT ---
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define a dummy decorator if numba is not available, so the code doesn't crash
    def njit(parallel=False):
        def decorator(func):
            return func
        return decorator
    prange = range


################################################################################
############################ SPIC Method  ######################################
################################################################################

# =============================================================================
# CALCULATION ENGINE #1: NUMBA-OPTIMIZED VERSION
# =============================================================================

@njit
def _solve_linear_regression_numba(X, y):
    """Solves a linear regression using Numba."""
    # Check if the matrix is singular to avoid errors
    if np.linalg.cond(X.T @ X) > 1e15:
        return np.inf, np.full(X.shape[1], np.nan), np.full(y.shape, np.nan)
    
    coeffs = np.linalg.solve(X.T @ X, X.T @ y)
    y_pred = X @ coeffs
    sse = np.sum((y - y_pred)**2)
    return sse, coeffs, y_pred

def _fit_model_numba(x_obs_sorted, y_obs_sorted, num_crossovers, min_points_per_segment=3):
    """Fits piecewise regression models using the Numba engine."""
    n_points = len(x_obs_sorted)

    if num_crossovers == 0:
        X_design = np.ones((n_points, 2))
        X_design[:, 1] = x_obs_sorted
        sse, _, y_pred = _solve_linear_regression_numba(X_design, y_obs_sorted)
        if np.isinf(sse): return None
        return {'best_taus': [], 'min_sse': sse, 'y_pred': y_pred}

    # Validations
    if n_points < min_points_per_segment * (num_crossovers + 1): return None
    unique_x_values = np.unique(x_obs_sorted)
    if len(unique_x_values) < num_crossovers + 2: return None
    potential_tau_values = unique_x_values[1:-1]
    if len(potential_tau_values) < num_crossovers: return None

    min_sse_val, best_taus_list, best_y_pred = float('inf'), None, None

    for current_taus_combination in itertools.combinations(potential_tau_values, num_crossovers):
        current_taus = sorted(list(current_taus_combination))
        
        # Segmentation validation
        valid_segmentation = True
        if np.sum(x_obs_sorted <= current_taus[0]) < min_points_per_segment: valid_segmentation = False
        if valid_segmentation:
            for i in range(num_crossovers - 1):
                if np.sum((x_obs_sorted > current_taus[i]) & (x_obs_sorted <= current_taus[i+1])) < min_points_per_segment:
                    valid_segmentation = False; break
        if valid_segmentation and np.sum(x_obs_sorted > current_taus[-1]) < min_points_per_segment: valid_segmentation = False
        if not valid_segmentation: continue

        X_design = np.ones((n_points, num_crossovers + 2))
        X_design[:, 1] = x_obs_sorted
        for i, tau_val in enumerate(current_taus):
            X_design[:, i + 2] = np.maximum(0, x_obs_sorted - tau_val)
        
        sse, _, y_pred = _solve_linear_regression_numba(X_design, y_obs_sorted)
        if sse < min_sse_val:
            min_sse_val, best_taus_list, best_y_pred = sse, current_taus, y_pred
            
    if best_taus_list is None: return None
    return {'best_taus': best_taus_list, 'min_sse': min_sse_val, 'y_pred': best_y_pred}

# =============================================================================
# CALCULATION ENGINE #2: SKLEARN-BASED VERSION (NO NUMBA)
# =============================================================================

def _fit_model_sklearn(x_obs_sorted, y_obs_sorted, num_crossovers, min_points_per_segment=3):
    """Fits piecewise regression models using the Scikit-learn engine."""
    n_points = len(x_obs_sorted)

    if num_crossovers == 0:
        model = LinearRegression()
        x_reshaped = x_obs_sorted.reshape(-1, 1)
        model.fit(x_reshaped, y_obs_sorted)
        y_pred = model.predict(x_reshaped)
        sse = np.sum((y_obs_sorted - y_pred)**2)
        return {'best_taus': [], 'min_sse': sse, 'y_pred': y_pred}
    
    # Validations (identical to Numba version)
    if n_points < min_points_per_segment * (num_crossovers + 1): return None
    unique_x_values = np.unique(x_obs_sorted)
    if len(unique_x_values) < num_crossovers + 2: return None
    potential_tau_values = unique_x_values[1:-1]
    if len(potential_tau_values) < num_crossovers: return None
    
    min_sse_val, best_taus_list, best_y_pred = float('inf'), None, None
    
    for current_taus_combination in itertools.combinations(potential_tau_values, num_crossovers):
        current_taus = sorted(list(current_taus_combination))
        
        # Segmentation validation (identical to Numba version)
        valid_segmentation = True
        if np.sum(x_obs_sorted <= current_taus[0]) < min_points_per_segment: valid_segmentation = False
        if valid_segmentation:
            for i in range(num_crossovers - 1):
                if np.sum((x_obs_sorted > current_taus[i]) & (x_obs_sorted <= current_taus[i+1])) < min_points_per_segment:
                    valid_segmentation = False; break
        if valid_segmentation and np.sum(x_obs_sorted > current_taus[-1]) < min_points_per_segment: valid_segmentation = False
        if not valid_segmentation: continue

        X_design_list = [x_obs_sorted]
        for tau_val in current_taus:
            X_design_list.append(np.maximum(0, x_obs_sorted - tau_val))
        X_design = np.vstack(X_design_list).T
        
        model = LinearRegression()
        model.fit(X_design, y_obs_sorted)
        y_pred = model.predict(X_design)
        sse = np.sum((y_obs_sorted - y_pred)**2)
        
        if sse < min_sse_val:
            min_sse_val, best_taus_list, best_y_pred = sse, current_taus, y_pred
            
    if best_taus_list is None: return None
    return {'best_taus': best_taus_list, 'min_sse': min_sse_val, 'y_pred': best_y_pred}

# =============================================================================
# PERMUTATION AND SEARCH LOGIC (ENGINE-AGNOSTIC)
# =============================================================================

def _process_single_permutation(perm_id, residuals_k0, y_fitted_k0, x_sorted, k0, k1, min_points, fit_function):
    """Processes a single permutation using the provided fitting function."""
    y_permuted = y_fitted_k0 + np.random.permutation(residuals_k0)
    
    results_k0_perm = fit_function(x_sorted, y_permuted, k0, min_points)
    if not results_k0_perm: return None
    
    results_k1_perm = fit_function(x_sorted, y_permuted, k1, min_points)
    if not results_k1_perm: return None
    
    sse_k0_perm, sse_k1_perm = results_k0_perm['min_sse'], results_k1_perm['min_sse']
    
    return (sse_k0_perm / sse_k1_perm) if sse_k1_perm != 0 else (float('inf') if sse_k0_perm > 0 else 1.0)


def perform_permutation_test_mp(x_obs_sorted, y_obs_sorted, k0, k1, fit_cache, fit_function, 
                                num_permutations, min_points_per_segment, significance_level, n_jobs):
    """Performs the permutation test, dispatching to the correct fitting engine."""
    
    # Fit models to original data using the selected fit function
    cache_key_k0 = (k0, min_points_per_segment)
    if cache_key_k0 not in fit_cache:
        fit_cache[cache_key_k0] = fit_function(x_obs_sorted, y_obs_sorted, k0, min_points_per_segment)
    results_k0_obs = fit_cache[cache_key_k0]
    if not results_k0_obs: return None

    cache_key_k1 = (k1, min_points_per_segment)
    if cache_key_k1 not in fit_cache:
        fit_cache[cache_key_k1] = fit_function(x_obs_sorted, y_obs_sorted, k1, min_points_per_segment)
    results_k1_obs = fit_cache[cache_key_k1]
    if not results_k1_obs: return None
    
    sse_k0_obs, sse_k1_obs = results_k0_obs['min_sse'], results_k1_obs['min_sse']
    y_fitted_k0_obs = results_k0_obs['y_pred']
    residuals_k0_obs = y_obs_sorted - y_fitted_k0_obs
    
    t_observed = (sse_k0_obs / sse_k1_obs) if sse_k1_obs != 0 else (float('inf') if sse_k0_obs > 0 else 1.0)
    
    starmap_args = [(i, residuals_k0_obs, y_fitted_k0_obs, x_obs_sorted, k0, k1, min_points_per_segment, fit_function) for i in range(num_permutations)]
    
    num_cores = os.cpu_count() if n_jobs == -1 else n_jobs
    if num_cores > 1:
        with multiprocessing.Pool(processes=num_cores) as pool:
            perm_results = pool.starmap(_process_single_permutation, starmap_args)
    else:
        perm_results = [_process_single_permutation(*args) for args in starmap_args]
            
    t_permuted_values = np.array([res for res in perm_results if res is not None])
    p_value = np.sum(t_permuted_values >= t_observed) / len(t_permuted_values) if len(t_permuted_values) > 0 else 1.0
    
    return {'reject_h0': p_value < significance_level, 'p_value': p_value}

# =============================================================================
# MAIN (PUBLIC) FUNCTION
# =============================================================================

def SPIC(x_obs, y_obs, max_k_to_test=3, num_permutations=200, 
                         min_points_per_segment=3, significance_level=0.05, 
                         n_jobs=-1, use_numba=True):
    """
    Finds the best number of crossovers (K) using a sequential permutation test.
    SPIC: Sequential Permutation for Identifying Crossovers. 
    Allows selecting between a Numba-optimized engine or a Scikit-learn based one.

    Parameters:
    -----------
    x_obs, y_obs : array_like
        The observed independent and dependent variables.
    max_k_to_test : int, optional
        The maximum number of crossovers (K) to test for. Default is 3.
    num_permutations : int, optional
        Number of permutations for the significance test. Default is 200.
    min_points_per_segment : int, optional
        Minimum number of data points required in each linear segment. Default is 3.
    significance_level : float, optional
        The alpha level for the permutation test. Default is 0.05.
    n_jobs : int, optional
        Number of CPU cores to use for parallel permutations. -1 means all available cores. Default is -1.
    use_numba : bool, optional
        If True (default), tries to use the Numba-optimized engine. If Numba is not
        available, it will automatically switch to False.
        If False, uses the Scikit-learn based engine.
    
    Returns:
    --------
    list
        A list of the indices (in the sorted array) where the crossovers occur.
        Returns an empty list if K=0 is the best model.

    Notes
    -----
    This function implements the sequential hypothesis testing method using
    permutations, as described in [1], to determine the optimal number of
    crossovers in a piecewise linear regression model. The core idea is to
    sequentially test a model with K crossovers against a model with K+1
    crossovers.

    References
    ----------
    [1] Ge, E., & Leung, Y. (2012). Detection of crossover time scales in multifractal detrended fluctuation analysis.
      Journal of Geographical Systems, 15(2), 115–147. doi:10.1007/s10109-012-0169-9

    """
    if use_numba and not NUMBA_AVAILABLE:
        warnings.warn("Numba was requested but is not installed. Falling back to the Scikit-learn engine.")
        use_numba = False
    
    engine_name = "Numba" if use_numba else "Scikit-learn"
    fit_function = _fit_model_numba if use_numba else _fit_model_sklearn
    
    logger.info(f"Starting SPIC search for best K (up to K={max_k_to_test}) using engine: {engine_name}.")
    
    # Search logic (engine-agnostic)
    x_obs, y_obs = np.asarray(x_obs), np.asarray(y_obs)
    sorted_indices = np.argsort(x_obs)
    x_obs_sorted, y_obs_sorted = x_obs[sorted_indices], y_obs[sorted_indices]
    
    fit_cache = {} 
    best_k = 0
    
    for k1_to_test in range(1, max_k_to_test + 1):
        k0_to_test = best_k
        logger.debug(f"\nTesting if K={k1_to_test} is better than K={k0_to_test}...")
        test_result = perform_permutation_test_mp(
            x_obs_sorted, y_obs_sorted, k0_to_test, k1_to_test, fit_cache, fit_function,
            num_permutations, min_points_per_segment, significance_level, n_jobs
        )
        if not test_result:
            logger.debug(f"Test K={k0_to_test} vs K={k1_to_test} failed. Stopping.")
            break
            
        if test_result['reject_h0']:
            logger.debug(f"Result: K={k1_to_test} is significantly better (p={test_result['p_value']:.4f}).")
            best_k = k1_to_test
        else:
            logger.debug(f"Result: K={k1_to_test} is NOT better (p={test_result['p_value']:.4f}).")
            logger.debug(f"Best K determined: K={best_k}.")
            break
            
    # Get the final Taus from the best model
    cache_key_best_k = (best_k, min_points_per_segment)
    if cache_key_best_k not in fit_cache:
        fit_cache[cache_key_best_k] = fit_function(x_obs_sorted, y_obs_sorted, best_k, min_points_per_segment)
    
    best_model_results = fit_cache.get(cache_key_best_k)
    best_k_taus_valores = best_model_results['best_taus'] if best_model_results else []

    logger.info(f"Search finished. Number of crossovers with most evidence: K = {best_k}")
    if not best_k_taus_valores:
        logger.info("Crossovers (x values): None (K=0)")
        return []
        
    logger.info(f"Crossovers (x values): {best_k_taus_valores}")
    crossover_indices = [np.where(x_obs_sorted == tau_val)[0][0] for tau_val in best_k_taus_valores]
    logger.info(f"Crossover indices (in x_obs_sorted): {crossover_indices}")
    
    return crossover_indices

################################################################################
############################ CDVA Method  ######################################
################################################################################

# =============================================================================
# CALCULATION ENGINE #1: NUMBA-OPTIMIZED VERSION
# =============================================================================

@njit
def _linear_fit_slope_numba(x, y):
    """
    Calculates only the slope of a linear fit y = mx + c.
    Optimized for Numba.
    """
    n = len(x)
    if n < 2:
        return np.nan
    
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x_sq = np.sum(x**2)
    
    denominator = (n * sum_x_sq - sum_x**2)
    if denominator == 0:
        return 0.0 if np.all(y == y[0]) else np.inf
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    return slope

@njit(parallel=True)
def _slope_dif_matrix_numba_internal(logS_arr, logFq_single_q_arr, N_s, w_min, w_arr):
    """
    Numba-optimized and internally parallelized version for a single q-moment.
    Calculates the slope difference matrix for a single logFq(q) series.
    """
    N_w = len(w_arr)
    N_s_B = N_s - 2 * (w_min - 1)
    if N_s_B <= 0:
        return np.empty((N_w, 0), dtype=np.float64)

    nu_single_q = np.zeros((N_w, N_s_B), dtype=np.float64)

    for ss in prange(N_s_B): # Numba's prange enables parallelism
        index_in_logS = ss + (w_min - 1)
        
        for ww in range(N_w):
            window_val = w_arr[ww]

            # Left-side linear fit
            start_left = max(0, index_in_logS - window_val + 1)
            x_left = logS_arr[start_left : index_in_logS + 1]
            y_left = logFq_single_q_arr[start_left : index_in_logS + 1]
            p_left_slope = _linear_fit_slope_numba(x_left, y_left)

            # Right-side linear fit
            end_right = min(N_s, index_in_logS + window_val)
            x_right = logS_arr[index_in_logS : end_right]
            y_right = logFq_single_q_arr[index_in_logS : end_right]
            p_right_slope = _linear_fit_slope_numba(x_right, y_right)
            
            nu_single_q[ww, ss] = np.abs(p_left_slope - p_right_slope)
            
    return nu_single_q

def _slope_dif_matrix_numba(logS, logFq, q, method, q_column):
    """Python wrapper that calls the Numba-optimized function for each q."""
    current_logFq = np.asarray(logFq, dtype=np.float64)
    current_q = np.asarray(q, dtype=np.float64)

    if method == 2:
        if q_column is None:
            q_column = np.argmin(np.abs(current_q - 2.0))
        current_logFq = current_logFq[:, q_column].reshape(-1, 1)
        current_q = np.array([current_q[q_column]])

    N_q_processed = current_q.shape[0]
    N_s_val = len(logS)
    w_min, w_max_val = 2, N_s_val - 1
    
    if w_max_val < w_min: return np.empty((0, 0))
    
    w_arr = np.arange(w_min, w_max_val + 1, dtype=np.int_)
    N_s_B_len = N_s_val - 2 * (w_min - 1)
    if N_s_B_len <= 0: return np.empty((len(w_arr), 0))
    
    all_nu = np.zeros((len(w_arr), N_s_B_len, N_q_processed), dtype=np.float64)

    for qq in range(N_q_processed):
        logFq_single_q = np.ascontiguousarray(current_logFq[:, qq])
        all_nu[:, :, qq] = _slope_dif_matrix_numba_internal(np.asarray(logS), logFq_single_q, N_s_val, w_min, w_arr)
    
    return np.nanmean(all_nu, axis=2)


# =============================================================================
# CALCULATION ENGINE #2: PLAIN PYTHON/NUMPY VERSION
# =============================================================================

def _slope_dif_matrix_plain(logS, logFq, q, method, q_column):
    """Calculates the M matrix using plain NumPy without Numba optimization."""
    logS_np, logFq_np, q_np = np.asarray(logS), np.asarray(logFq), np.asarray(q)
    
    if method == 2:
        if q_column is None:
            q_column = np.argmin(np.abs(q_np - 2.0))
        logFq_np = logFq_np[:, q_column].reshape(-1, 1)
        q_np = np.array([q_np[q_column]])
    
    N_q, N_s = len(q_np), len(logS_np)
    w_min, w_max = 2, N_s - 1
    if w_max < w_min: return np.empty((0,0))
    
    w = np.arange(w_min, w_max + 1)
    N_w = len(w)
    N_s_B = N_s - 2 * (w_min - 1)
    if N_s_B <= 0: return np.empty((N_w, 0))

    nu = np.zeros((N_w, N_s_B, N_q))

    for qq in range(N_q):
        for ss in range(N_s_B):
            index = ss + (w_min - 1)
            for ww in range(N_w):
                window = w[ww]
                
                start_left = max(0, index - window + 1)
                x_left = logS_np[start_left:index+1]
                y_left = logFq_np[start_left:index+1, qq]
                p_left = np.polyfit(x_left, y_left, 1)

                end_right = min(N_s, index + window)
                x_right = logS_np[index:end_right]
                y_right = logFq_np[index:end_right, qq]
                p_right = np.polyfit(x_right, y_right, 1)
                
                nu[ww, ss, qq] = abs(p_left[0] - p_right[0])
    
    return np.mean(nu, axis=2)

# =============================================================================
# COMMON LOGIC (FIND_CROSSOVER)
# =============================================================================

def find_crossover(M):
    """Finds the crossover in the slope difference matrix M."""
    if not isinstance(M, np.ndarray) or M.size == 0 or M.shape[0] < 2 or M.shape[1] < 2:
        warnings.warn("(CDVA/find_crossover): M matrix is empty or too small for analysis.")
        return np.nan, np.nan, 0, np.array([])
    if np.all(np.isnan(M)):
        warnings.warn("(CDVA/find_crossover): M matrix contains only NaNs.")
        return np.nan, np.nan, 0, np.array([])

    # (Using the robust logic from your CDVAp.py)
    var_col = np.nanvar(M, axis=0, ddof=1)
    var_row = np.nanvar(M, axis=1, ddof=1)

    dif = np.diff(var_row[np.isfinite(var_row)])
    index_dif = np.where(dif > 0)[0]
    i_min = index_dif[0] + 1 if len(index_dif) > 0 else 0
    
    var_row_1 = var_row[i_min:]
    if var_row_1.size == 0 or np.all(np.isnan(var_row_1)): return np.nan, np.nan, 0, np.array([])
    
    i_max = i_min + np.nanargmax(var_row_1)
    range_row_2 = np.arange(0, i_max + 1)
    dif_2 = np.diff(var_row[range_row_2])
    index_dif_2 = np.where(dif_2 < 0)[0]
    i_cut = index_dif_2[-1] + 1 if len(index_dif_2) > 0 else 0

    var_col_cut = np.nanvar(M[i_cut:, :], axis=0, ddof=1)
    if np.all(np.isnan(var_col_cut)): return np.nan, np.nan, i_cut, np.array([])
    
    peaks, _ = find_peaks(np.nan_to_num(var_col_cut, nan=-np.inf))
    if len(peaks) == 0: return np.nan, np.nan, i_cut, np.array([])

    pks_values = var_col_cut[peaks]
    valid_peaks = peaks[~np.isnan(pks_values)]
    if len(valid_peaks) == 0: return np.nan, np.nan, i_cut, np.array([])

    sorted_peaks = valid_peaks[np.argsort(var_col_cut[valid_peaks])[::-1]]
    i1 = sorted_peaks[0]
    tol = 10 * np.nanmin(var_col_cut)
    
    valley = None
    for peak in sorted_peaks[1:]:
        a, b = min(peak, i1), max(peak, i1)
        segment = var_col_cut[a : b + 1]
        candidate_valley = np.where(segment <= tol)[0]
        if candidate_valley.size > 0:
            valley = candidate_valley + a
            break
            
    if valley is None or valley.size == 0: return np.nan, np.nan, i_cut, np.array([])
    
    if valley.size == 1:
        index_s_cross = valley[0]
        slope_dif_mean = np.nanmean(np.abs(M[:, index_s_cross - 1])) if index_s_cross > 0 else np.nan
    else:
        M_sel = M[i_cut:, valley]
        col_sums = np.nansum(np.abs(M_sel), axis=0)
        norm_1_idx = np.nanargmax(col_sums)
        slope_dif_mean = np.nanmean(np.abs(M_sel[:, norm_1_idx]))
        index_s_cross = valley[norm_1_idx]
        
    return index_s_cross, slope_dif_mean, i_cut, valley

# =============================================================================
# MAIN (PUBLIC) FUNCTION WITH ENGINE SELECTOR
# =============================================================================

def CDVA(logS, logFq, q, method=1, q_column=None, use_numba=False):
    """Crossover Detection based on Variance of slope Differences (CDV-A).

    This function implements the CDV-A algorithm [2] to find the most
    prominent crossover point in a log-log plot of fluctuation functions
    Fq(s) vs. scales s.

    Parameters
    ----------
    logS : array_like
        1D array of the logarithm of the scales.
    logFq : array_like
        2D array of the logarithm of the fluctuation functions.
        Rows correspond to scales, columns to q-moments.
    q : array_like
        1D array of the q-moments.
    method : {1, 2}, optional
        Method to use: 1 for averaging over all q-moments, 2 for using
        only q=2. Default is 1.
    q_column : int, optional
        0-based index of the column to use in logFq when method=2.
        If None, the column closest to q=2 is found automatically.
    use_numba : bool, optional
        If True, attempts to use the Numba-optimized engine for speed. 
        If Numba is not installed, it will fall back to the plain NumPy version.
        Defaults to False.
    
    Returns
    -------
    tuple
        A tuple containing:
        - index_s_cross (int): 0-based index of the crossover in the logS array.
        - slope_dif_mean (float): Mean of the slope differences at the crossover.
        - i_cut (int): Row index used to trim noise-affected variances.
        - valley (ndarray): Array of column indices forming the detected valley.

    Notes
    -----
    The CDV-A method identifies potential crossover regions by analyzing the
    variance of the differences between left-side and right-side log-log
    slopes, computed across multiple window sizes. This implementation is
    based on the description and MATLAB code provided in [1].

    References
    ----------
    [2] Moreno-Pulido, S., de la Torre, J.C., Ruiz, P. et al. Crossover
        detection based on variances of slope differences for multi-fractal
        detrended fluctuation analysis (MF-DFA). Nonlinear Dyn 113, 
        7425–7457 (2025). https://doi.org/10.1007/s11071-024-10478-1"""
    engine_func = _slope_dif_matrix_plain
    if use_numba:
        if NUMBA_AVAILABLE:
            engine_func = _slope_dif_matrix_numba
        else:
            warnings.warn("Numba was requested but is not installed. Using the plain NumPy version.")
    
    # Calculate the slope difference matrix using the selected engine
    M = engine_func(logS, logFq, q, method, q_column)

    # Find the crossover (this logic is the same for both)
    if not isinstance(M, np.ndarray) or M.size == 0:
        warnings.warn("(CDVA): Calculated M matrix is invalid or empty.")
        return np.nan, np.nan, 0, np.array([])
        
    return find_crossover(M)