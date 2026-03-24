"""
Created on Wed Jun 18 12:49:42 2025

@author: Nahuel Mendez & Sebastian Jaroszewicz
"""

import numpy as np
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
import os
import warnings,logging
logger = logging.getLogger(__name__)


# MFDFA (Multifractal Detrended Fluctuation Analysis) - Auxiliary function
def _process_scale(s_val, profile_data, q_values_arr, poly_order, N_len, process_from_end=False):
    """
    Processes a single scale 's' for MFDFA calculation using vectorized matrix operations.

    This function divides the profile data into segments of length 's_val', performs 
    polynomial detrending on all segments simultaneously using a least-squares 
    solution, and computes the fluctuation function F_q for all values in q_values_arr.

    Parameters:
    -----------
    s_val : int
        The current scale (window size) to process.
    profile_data : ndarray
        The integrated profile of the time series (1D array).
    q_values_arr : ndarray
        1D array of q exponents to compute the multifractal spectrum.
    poly_order : int
        Order of the polynomial for local detrending (e.g., 1 for linear).
    N_len : int
        Total length of the profile_data.
    process_from_end : bool, optional
        If True, processes segments from both the beginning and the end of the 
        series (2Ns segments total), ensuring no data is lost. Default is False.

    Returns:
    --------
    tuple: (int, ndarray)
        A tuple containing:
        - s_val: The processed scale.
        - F_q_for_this_s: 1D array of shape (len(q_values_arr),) containing 
          the fluctuation values for each q.

   """
    if s_val <= poly_order: # Evitamos errores de grado de polinomio
        return s_val, np.zeros(len(q_values_arr))

    # 1. Preparar segmentos de forma matricial (Hacia adelante)
    num_segments = N_len // s_val
    # Cortamos la serie para que sea divisible por s_val y hacemos reshape
    # shape: (num_segments, s_val)
    segments_fwd = profile_data[:num_segments * s_val].reshape(num_segments, s_val)
    
    # 2. Preparar segmentos (Hacia atrás)
    if process_from_end:
        segments_bwd = profile_data[N_len - num_segments * s_val:].reshape(num_segments, s_val)
        all_segments = np.vstack([segments_fwd, segments_bwd])
    else:
        all_segments = segments_fwd

    # 3. DETRENDING VECTORIZADO (El "Modo Rápido")
    # Creamos una sola matriz de Vandermonde para todos los segmentos
    x = np.arange(s_val)
    A = np.vander(x, poly_order + 1) # Matriz base para el ajuste
    
    # Resolvemos los coeficientes para TODOS los segmentos a la vez
    # A @ Coeffs = all_segments.T
    coeffs, _, _, _ = np.linalg.lstsq(A, all_segments.T, rcond=None)
    
    # Calculamos la tendencia y restamos
    trends = (A @ coeffs).T
    detrended = all_segments - trends
    
    # 4. Cálculo de Varianzas en bloque
    # axis=1 calcula la media de cada segmento (fila)
    segment_variances = np.mean(detrended**2, axis=1)
    
    # Filtramos varianzas cero para evitar logs de números negativos o ceros
    segment_variances = segment_variances[segment_variances > 0]
    
    if len(segment_variances) == 0:
        return s_val, np.zeros(len(q_values_arr))

    # 5. Cálculo de F_q (Vectorizado también para q)
    F_q_for_this_s = np.zeros(len(q_values_arr))
    
    # Para q != 0 (Usamos broadcasting de numpy)
    q_non_zero = q_values_arr != 0
    q_vals = q_values_arr[q_non_zero][:, np.newaxis] # Preparamos para operar con matrices
    
    # F_q = [mean(var^(q/2))]^(1/q)
    f_q_vals = np.mean(segment_variances**(q_vals / 2.0), axis=1)**(1.0 / q_vals.flatten())
    F_q_for_this_s[q_non_zero] = f_q_vals
    
    # Para q == 0 (Caso especial: Exponente del promedio de logs)
    if np.any(q_values_arr == 0):
        q_zero_idx = np.where(q_values_arr == 0)[0][0]
        F_q_for_this_s[q_zero_idx] = np.exp(0.5 * np.mean(np.log(segment_variances)))

    return s_val, F_q_for_this_s

#-----------------------------------------------------------------------------------------------------#
#--------MFDFA (Multifractal Detrended Fluctuation Analysis) in parallel using multiprocessing--------#
#-----------------------------------------------------------------------------------------------------#
def mfdfa(data, q_values, scales, order=1, num_cores=None,
          segments_from_both_ends=False,scale_range_for_hq=None,validate=True):
    """
    Performs Multifractal Detrended Fluctuations Analysis (MFDFA) in parallel.

    Parameters:
    -----------
    data : array_like
        The time series to analyze (one-dimensional).
    q_values : array_like
        The range of q moments for the analysis.
    scales : array_like
        The scales (segment lengths) to consider. Must be integers.
    order : int, optional
        The order of the polynomial for detrending (default is 1, linear).
    num_cores : int, optional
        Number of CPU cores to use. If None, use os.cpu_count().
    segments_from_both_ends : bool, optional
        If True, segments are taken from the start and end of the series.
        If False (default) segments are taken only from the start.
    scale_range_for_hq : tuple or list, optional
        Tuple (min_s, max_s) defines the scale range to be used to calculate 
        the exponent h(q). If None (default), all valid scales are used.
    validate: bool, optional
        If True (default), theoretical and concavity masks are applied to validate numerical results
        If False, numerical values are returned without a validation step. 

    Return:
    --------
    q   : ndarray
        q-values or moment exponents
    h_q : ndarray
        The generalized Hurst exponent for each value of q.
    tau_q : ndarray
        The mass scaling function for each value of q.
    alpha : ndarray
        The singularity (or Hölder) exponent.
    f_alpha : ndarray
        The singularity spectrum.
    F_q_s : ndarray
        The fluctuation function F_q(s) for each q and s..
    """
    # =========================================================================
    # --- CHECKS AND SETTINGS ---
    # =========================================================================
    data_arr = np.asarray(data)
    q_values_arr = np.asarray(q_values)
    scales_arr = np.asarray(scales, dtype=int)
    
    #.Check the s_min
    if np.min(scales_arr)<order+1:
        scales_arr=scales_arr[scales_arr>=order+1]
        warnings.warn("min(scale) is less than m+1. Using a safe min_scale")
        
    # Compute the integrated profile of the time series
    profile_data = np.cumsum(data_arr - np.mean(data_arr))
    N_len = len(profile_data)

    if num_cores is None:
        # Attempt to get CPU count, default to 1 if os.cpu_count() returns None
        num_cores = os.cpu_count() if os.cpu_count() is not None else 1
    
    num_cores = min(num_cores, len(scales_arr), os.cpu_count() if os.cpu_count() else 1)

    # Prepare arguments for each task including  'segments_from_both_ends'
    tasks = [(s_val, profile_data, q_values_arr, order, N_len, segments_from_both_ends) for s_val in scales_arr]
   
    # =========================================================================
    # --- COMPUTING F_q(s) ---
    # =========================================================================
    F_q_s_matrix = np.full((len(q_values_arr), len(scales_arr)), np.nan) 
    
    if num_cores == 1:
        # sequential mode
        results_list = [_process_scale(*task) for task in tasks]
        
    else:
        # parallel mode
        with ThreadPoolExecutor(max_workers=num_cores) as executor:
            results_list = list(executor.map(lambda t: _process_scale(*t), tasks))

    scale_to_original_idx = {s_val: i for i, s_val in enumerate(scales_arr)}
    for s_processed, F_q_for_s_processed_arr in results_list:
        if s_processed in scale_to_original_idx:
            original_idx = scale_to_original_idx[s_processed]
            F_q_s_matrix[:, original_idx] = F_q_for_s_processed_arr
    
    # =========================================================================
    # ---  COMPUTING h(q) ---
    # =========================================================================
    
    valid_scales_mask = np.any(np.isfinite(F_q_s_matrix) & (F_q_s_matrix > 0), axis=0)
    if not np.any(valid_scales_mask):
        warnings.warn("All fluctuacion functions F_q(s) are zero or Nan. Cannot calculate exponents.")
        nan_res = np.full(len(q_values_arr), np.nan)
        return q_values_arr,nan_res, nan_res, nan_res, nan_res, F_q_s_matrix

    F_q_s_filtered = F_q_s_matrix[:, valid_scales_mask]
    scales_filtered = scales_arr[valid_scales_mask]
    
    if len(scales_filtered) < 2:
        warnings.warn("There is no enough valid scales for fitting. At least 2 is needed.")
        nan_res = np.full(len(q_values_arr), np.nan)
        return q_values_arr, nan_res, nan_res, nan_res, nan_res, F_q_s_matrix
    
    #.Initialize array with Nan
    h_q_arr = np.full(len(q_values_arr), np.nan)
    
    for i_q_idx in range(len(q_values_arr)):
        Fqs_to_fit = F_q_s_filtered[i_q_idx, :]
        scales_to_fit = scales_filtered
        
        # Apply the range for h(q) fit if its given
        if scale_range_for_hq is not None and isinstance(scale_range_for_hq, (list, tuple)) and len(scale_range_for_hq) == 2:
            min_s_fit, max_s_fit = scale_range_for_hq
            fit_region_mask = (scales_filtered >= min_s_fit) & (scales_filtered <= max_s_fit)
            #.Apply mask to F(q,s)
            scales_to_fit = scales_filtered[fit_region_mask]
            Fqs_to_fit = F_q_s_filtered[i_q_idx, :][fit_region_mask]
           
        valid_points_mask = np.isfinite(Fqs_to_fit) & (Fqs_to_fit > 0)
        scales_final_for_fit = scales_to_fit[valid_points_mask]
        Fqs_final_for_fit = Fqs_to_fit[valid_points_mask]
        # Check if we have got enough values for a linear fit
        if len(scales_final_for_fit) < 2:
            continue  # If not, continue to the next q
    
        log_scales = np.log(scales_final_for_fit)
        log_Fqs = np.log(Fqs_final_for_fit)
            
        try:
            fit = stats.linregress(log_scales, log_Fqs, alternative='two-sided')
            h_q_arr[i_q_idx] = fit.slope
            #.Assess the quality of the linear fit
            if fit.rvalue**2<0.85 or fit.pvalue>0.05:
                #Analyze residuals
                linear_fit =fit.slope*log_scales+fit.intercept
                fit_residuals=log_Fqs-linear_fit
                shapiro_test=stats.shapiro(fit_residuals)
                if shapiro_test.pvalue<0.05:  #.Significance of 5%
                    warnings.warn(f"Not good fit in h(q) for q={q_values_arr[i_q_idx]}",RuntimeWarning)
        
        except ValueError as e:
            warnings.warn(f"Could not calculate h(q) for q={q_values_arr[i_q_idx]:.2f} due to a regression error. This point will be skipped.", RuntimeWarning)
            logger.error(f"Linear regression failed for q={q_values_arr[i_q_idx]:.2f}.", exc_info=True)
            continue
        
            
    # =========================================================================
    # ---  COMPUTING MULTIFRACTAL PARAMETERS ---
    # =========================================================================
    tau_q_arr = q_values_arr * h_q_arr - 1

    valid_hq_mask = ~np.isnan(h_q_arr) 
    if not np.any(valid_hq_mask) or len(q_values_arr[valid_hq_mask]) < 2:
        warnings.warn("It was not possible to calculate h(q) for sufficient q values. The multifractal spectrum cannot be determined.")
        nan_alpha_res = np.full(len(q_values_arr), np.nan)
        return q_values_arr, h_q_arr, tau_q_arr, nan_alpha_res, nan_alpha_res, F_q_s_matrix

    q_filt = q_values_arr[valid_hq_mask]
    tau_q_filt = tau_q_arr[valid_hq_mask]

    if len(q_filt) < 2: 
        alpha_arr = np.full(len(q_values_arr), np.nan)
        f_alpha_arr = np.full(len(q_values_arr), np.nan)
    else:
        alpha_filt = np.gradient(tau_q_filt, q_filt)
        alpha_arr = np.interp(q_values_arr, q_filt, alpha_filt, left=np.nan, right=np.nan)
        alpha_arr[np.isnan(h_q_arr)] = np.nan
        f_alpha_arr = q_values_arr * alpha_arr - tau_q_arr
        
    # =========================================================================
    # --- ASSESMENT OF NUMERICAL LIMITS ---
    # =========================================================================
    #. Singularity spectrum cant be negative or greater than 1. 
    #. Alpha must be positive
    if validate:
        initial_mask = (f_alpha_arr >= 0) & (alpha_arr > 0) & (f_alpha_arr <= 1)
        #.Find q=0 or nearest
        center_idx = np.argmin(np.abs(q_values))
        # Verify if is a valid point
        if not initial_mask[center_idx]:
            warnings.warn("The central point is not valid. Can't define a robust range.")
            return q_values_arr,nan_res, nan_res, nan_res, nan_res, F_q_s_matrix
        # Expand from center to the right
        valid_end_idx = center_idx
        while valid_end_idx + 1 < len(initial_mask) and initial_mask[valid_end_idx + 1]:
            valid_end_idx += 1
    
        # Expand from the left to the center
        valid_start_idx = center_idx
        while valid_start_idx - 1 >= 0 and initial_mask[valid_start_idx - 1]:
            valid_start_idx -= 1
            
        # Build theoretical mask
        theoretical_mask = np.zeros_like(initial_mask, dtype=bool)
        theoretical_mask[valid_start_idx : valid_end_idx + 1] = True
        
        #.Sigularity spectra cannot grow (from max to edges)
        if len(f_alpha_arr) < 3:
           return np.ones_like(f_alpha_arr, dtype=bool)
       
        # Start from the mid
        idx_mid = np.argmin(np.abs(alpha_arr - np.median(alpha_arr)))
        #.From max to the right
        idx_derecho = idx_mid
        while idx_derecho + 1 < len(f_alpha_arr) and f_alpha_arr[idx_derecho + 1] < f_alpha_arr[idx_derecho]:
            idx_derecho += 1
        #.From max to the left
        idx_izquierdo = idx_mid
        while idx_izquierdo - 1 >= 0 and f_alpha_arr[idx_izquierdo - 1] < f_alpha_arr[idx_izquierdo]:
            idx_izquierdo -= 1
            
        #.Build concavity mask
        concavity_mask = np.zeros_like(f_alpha_arr, dtype=bool)
        concavity_mask[idx_izquierdo : idx_derecho + 1] = True    
        
        #.Build final mask (valid only when both are True)
        final_mask = theoretical_mask & concavity_mask
        
        # =========================================================================
        # --- APPLY MASKS ---
        # =========================================================================
        
        if not np.any(final_mask):
            warnings.warn("No q-values valid for this scale range. Probably the series is not multifractal.")
            nan_res = np.full(len(q_values_arr), np.nan)
            # Return empty arrays
            return q_values_arr, nan_res, nan_res, nan_res, nan_res, F_q_s_matrix
        
        #.Trim arrays
        q_valid = q_values[final_mask]
        h_q_valid = h_q_arr[final_mask]
        tau_q_valid = tau_q_arr[final_mask]
        alpha_valid = alpha_arr[final_mask]
        f_alpha_valid = f_alpha_arr[final_mask]
        F_q_s_valid_matrix = F_q_s_matrix[final_mask, :]
        logger.info(f"q-values trimmed to: [{np.min(q_valid):.2f},{np.max(q_valid):.2f}]")
        return q_valid, h_q_valid, tau_q_valid, alpha_valid, f_alpha_valid, F_q_s_valid_matrix
    else:
        #.No validation filters applied 
        return q_values_arr, h_q_arr, tau_q_arr, alpha_arr, f_alpha_arr, F_q_s_matrix
    
    
#--Bootstrapping
def bootstrap_multifractal_parameters(scales, F_q_s, q_values, n_boot=1000, conf_interval=0.95):
    """
    Performs a fully vectorized bootstrap for MFDFA parameters (h, tau, alpha, f_alpha).
    Uses tensor linear algebra instead of for-loops for resampling.

    Parameters:
    -----------
    scales : array (N_scales,)
        The scales 's' used in the analysis.
    F_q_s : array (N_q, N_scales)
        Fluctuation function matrix (pre-filtered for NaNs/Zeros).
    q_values : array (N_q,)
        The q exponents.
    n_boot : int
        Number of bootstrap iterations.
    conf_interval : float
        Confidence level (e.g., 0.95 for 95% CI).

    Returns:
    --------
    dict
        Dictionary with keys 'h', 'tau', 'alpha', 'f_alpha'.
        Each value is a tuple: (mean, ci_lower, ci_upper, err_low, err_high).
    """
    # ---------------------------------------------------------
    # 1. DATA PREPARATION (Log-Log Domain)
    # ---------------------------------------------------------
    scales = np.asarray(scales)
    log_scales = np.log(scales)
    log_Fqs = np.log(F_q_s)

    N_q, N_s = log_Fqs.shape

    # ---------------------------------------------------------
    # 2. BASE OLS FIT & RESIDUALS
    # ---------------------------------------------------------
    # Design Matrix X: [log_scales, 1] -> Shape (N_s, 2)
    X = np.vstack([log_scales, np.ones(N_s)]).T

    # Pre-calculate the OLS Projector: (X^T X)^-1 X^T -> Shape (2, N_s)
    # This fixed geometry is what makes the bootstrap ultra-fast.
    try:
        OLS_projector = np.linalg.inv(X.T @ X) @ X.T
    except np.linalg.LinAlgError:
        warnings.warn("Singular matrix in Bootstrap OLS. Returning NaNs.")
        nan_arr = np.full(N_q, np.nan)
        return {k: (nan_arr, nan_arr, nan_arr, nan_arr, nan_arr) for k in ['h','tau','alpha','f_alpha']}

    # Vectorized original fit for all q values simultaneously
    # log_Fqs.T is (N_s, N_q). Resulting Betas are (2, N_q)
    Betas_orig = OLS_projector @ log_Fqs.T

    # Predictions and Residuals
    # Y_pred: (N_s, 2) @ (2, N_q) -> (N_s, N_q). Transposed to (N_q, N_s) to match input
    Y_pred = (X @ Betas_orig).T
    Residuals = log_Fqs - Y_pred

    # ---------------------------------------------------------
    # 3. TENSOR BOOTSTRAP (The Vectorized Magic)
    # ---------------------------------------------------------
    # Generate random indices for residual resampling with replacement
    # Shape: (N_s, n_boot)
    boot_indices = np.random.randint(0, N_s, size=(N_s, n_boot))

    # Construct the Bootstrap Residuals Tensor
    # Res_boot shape: (N_q, N_s, n_boot) using fancy indexing
    Res_boot = Residuals[:, boot_indices]

    # Add to original prediction (Broadcasting)
    # Expand Y_pred to (N_q, N_s, 1) for 3D addition
    Y_boot = Y_pred[:, :, np.newaxis] + Res_boot

    # ---------------------------------------------------------
    # 4. MASSIVE OLS SOLUTION
    # ---------------------------------------------------------
    # Flatten last two dimensions to solve all bootstraps in one matrix mult
    Y_boot_flat = Y_boot.transpose(1, 0, 2).reshape(N_s, -1) # Shape (N_s, N_q * n_boot)

    # Single matrix multiplication
    Betas_boot_flat = OLS_projector @ Y_boot_flat # Shape (2, N_q * n_boot)

    # Reshape back to (2, N_q, n_boot). Row 0 contains h(q) slopes.
    h_boot_dist = Betas_boot_flat[0, :].reshape(N_q, n_boot)

    # ---------------------------------------------------------
    # 5. ERROR PROPAGATION (Tau, Alpha, f_Alpha)
    # ---------------------------------------------------------
    # A. Tau(q) = q * h(q) - 1
    q_col = q_values[:, np.newaxis]
    tau_boot_dist = q_col * h_boot_dist - 1

    # B. Alpha(q) = d(tau)/dq
    # Compute gradient along axis 0 (q-axis) for each bootstrap column
    alpha_boot_dist = np.gradient(tau_boot_dist, q_values, axis=0)

    # C. f(alpha) = q * alpha - tau
    f_alpha_boot_dist = q_col * alpha_boot_dist - tau_boot_dist

    # ---------------------------------------------------------
    # 6. STATISTICS (Percentiles)
    # ---------------------------------------------------------
    alpha_level = 1.0 - conf_interval
    lp, up = (alpha_level / 2.0) * 100, (1.0 - alpha_level / 2.0) * 100

    results = {}

    def get_stats(dist_matrix):
        """Helper to compute mean and confidence intervals across bootstrap axis."""
        mean_val = np.mean(dist_matrix, axis=1)
        ci_low = np.percentile(dist_matrix, lp, axis=1)
        ci_high = np.percentile(dist_matrix, up, axis=1)
        
        # Absolute Error Bars (distance to mean) for plotting
        err_low = mean_val - ci_low
        err_high = ci_high - mean_val
        return mean_val, ci_low, ci_high, err_low, err_high

    results['h'] = get_stats(h_boot_dist)
    results['tau'] = get_stats(tau_boot_dist)
    results['alpha'] = get_stats(alpha_boot_dist)
    results['f_alpha'] = get_stats(f_alpha_boot_dist)

    return results
    