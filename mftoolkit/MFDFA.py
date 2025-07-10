import numpy as np
from scipy import stats
import multiprocessing
import os
import warnings,logging
logger = logging.getLogger(__name__)

# MFDFA (Multifractal Detrended Fluctuation Analysis) - Auxiliary function
def _process_scale(s_val, profile_data, q_values_arr, poly_order, N_len, process_from_end=False):
    """
    Processes a single scale 's' for MFDFA calculation.

    Parameters:
    -----------
    s_val : int
        The current scale to process.
    profile_data : ndarray
        The integrated profile of the time series.
    q_values_arr : ndarray
        Array of q values.
    poly_order : int
        Order of the polynomial for detrending.
    N_len : int
        Length of the profile (len(profile_data)).
    process_from_end : bool, optional
        If True, process segments from start and end.
        If False (default), processes only segments from the start.

    Return:
    --------
    tuple: (int, ndarray)
        The processed scale (s_val) and the array F_q for that scale.
    """
    if s_val == 0:
        return s_val, np.zeros(len(q_values_arr))

    num_segments_for_direction = N_len // s_val
    
    segment_variances_list = []

    # Segmentos desde el inicio (siempre se procesan)
    for v_idx in range(num_segments_for_direction):
        start_idx = v_idx * s_val
        end_idx = (v_idx + 1) * s_val
        if end_idx > N_len: 
            continue
        segment = profile_data[start_idx : end_idx]
        
        if len(segment) < poly_order + 1: 
            segment_variances_list.append(0) 
            continue

        time_axis_segment = np.arange(len(segment))
        
        coeffs = np.polyfit(time_axis_segment, segment, poly_order)
        trend = np.polyval(coeffs, time_axis_segment)
        detrended_segment = segment - trend
        
        segment_variances_list.append(np.mean(detrended_segment**2))

    if process_from_end:
        for v_idx in range(num_segments_for_direction):
            start_idx = N_len - (v_idx + 1) * s_val
            end_idx = N_len - v_idx * s_val
            if start_idx < 0: 
                continue
            segment = profile_data[start_idx : end_idx]

            if len(segment) < poly_order + 1:
                segment_variances_list.append(0)
                continue

            time_axis_segment = np.arange(len(segment)) 
            
            coeffs = np.polyfit(time_axis_segment, segment, poly_order)
            trend = np.polyval(coeffs, time_axis_segment)
            detrended_segment = segment - trend
            
            segment_variances_list.append(np.mean(detrended_segment**2))
    
    segment_variances_arr = np.array(segment_variances_list)
    segment_variances_arr = segment_variances_arr[segment_variances_arr > 0] 

    F_q_for_this_s = np.zeros(len(q_values_arr))
    if len(segment_variances_arr) == 0:
        return s_val, F_q_for_this_s 

    for i_q_idx, q_val_current in enumerate(q_values_arr):
        if q_val_current == 0:
            if np.any(segment_variances_arr <= 0): 
                 F_q_for_this_s[i_q_idx] = np.nan
            else:
                F_q_for_this_s[i_q_idx] = np.exp(0.5 * np.mean(np.log(segment_variances_arr)))
        else:
            F_q_for_this_s[i_q_idx] = np.mean(segment_variances_arr**(q_val_current / 2.0))**(1.0 / q_val_current)
            
    return s_val, F_q_for_this_s

#-----------------------------------------------------------------------------------------------------#
#--------MFDFA (Multifractal Detrended Fluctuation Analysis) in parallel using multiprocessing--------#
#-----------------------------------------------------------------------------------------------------#
def mfdfa(data, q_values, scales, order=1, num_cores=None,
          segments_from_both_ends=False,scale_range_for_hq=None):
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

    Return:
    --------
    q   : ndarray
        q-values valid for the time series after applying theoretical and concavity masks.
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
    
    pool = None
    try:
        pool = multiprocessing.Pool(processes=num_cores)
        results_list = pool.starmap(_process_scale, tasks)
    finally:
        if pool:
            pool.close() 
            pool.join()  

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
    
    initial_mask = (f_alpha_arr >= 0) & (alpha_arr > 0) & (f_alpha_arr <= 1)
    #.Find q=0 or nearest
    center_idx = np.argmin(np.abs(q_values))
    # Verify if is a valid point
    if not initial_mask[center_idx]:
        warnings.warn("The central point is not valid. Can't define a robust range.")
        return np.zeros_like(initial_mask, dtype=bool)
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
    