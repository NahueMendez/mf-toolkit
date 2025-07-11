# mf-toolkit

[![PyPI version](https://badge.fury.io/py/mf-toolkit.svg)](https://pypi.org/project/mf-toolkit/0.1.3/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15858623.svg)](https://doi.org/10.5281/zenodo.15858623)

A high-performance Python toolkit for the analysis of multifractal time series and complex systems.

This project provides robust and efficient implementations of key algorithms such as MFDFA, as well as methods for crossover detection and surrogate data generation for hypothesis testing.

## Main Features

* **Fast MFDFA**: Efficient implementation of MFDFA analysis.
* **Parallel Processing**: Optimized to utilize multiple CPU cores.
* **Dual Engine**: Uses *Numba* for maximum performance or a Scikit-learn backend for easy installation.
* **Crossover Detection**: Includes SPIC and CDVA methods to identify crossover changes.
* **Surrogate Generation**: Creates test data with IAAFT and Shuffling methods.
* **Validated Results**: Automatically filters and returns only the range of `q` where the multifractal parameters are mathematically plausible, avoiding numerical instabilities.

## Installation

You can install the stable version from PyPI:

```bash
pip install mf-toolkit
```

To install the performance-optimized version of Numba, use:
```bash
pip install mf-toolkit[numba]
```

## Example of Usage

Here is a simple example of how to use the main `mfdfa` function:

```python
import numpy as np
from mf_toolkit import mfdfa,SPIC
import multiprocessing
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    # 1. Generate an example multifractal series (binomial cascade)
    p=0.6
    n_points = 16384
    k = int(np.ceil(np.log2(n_points)))
    N = 2**k
    example_series = np.ones(N)
    for i in range(k):
        num_segments = 2**i
        segment_len = N // (2 * num_segments)
        new_series = np.copy(example_series)
        for j in range(num_segments):
            start_idx = j * 2 * segment_len
                
            # Half of the left: Multiply by p
            left_slice = slice(start_idx, start_idx + segment_len)
            new_series[left_slice] *= p
                
            # Half of the right: Multiply by (1-p)
            right_slice = slice(start_idx + segment_len, start_idx + 2 * segment_len)
            new_series[right_slice] *= (1 - p)
                
        example_series = new_series
            
    
    
    # 2. Define parameters for MFDFA
    
    #.q exponents
    q_values= np.linspace(-5, 5, 100)
    
    #.Scales
    min_scale_val = 200
    max_scale_val = 8000
    num_scales_val = 100 
    scales = np.floor(np.logspace(np.log10(min_scale_val), np.log10(max_scale_val), num_scales_val)).astype(int)
    scales = np.unique(scales) 
    
    #.Order of polynomial for local detrending
    detrend_order=1
    #.Using the whole dataset processing in both directions
    process_both=True
    #.Parallel computation
    num_cores_to_use = 4 
    
    #.Define a subrange for fitting (optional)
    if len(scales) >= 5: # We need some point for defining a subrange
        idx_start_fit = np.where(np.log(scales)>=np.log(300))[0][0]
        idx_end_fit = np.where(np.log(scales)<=np.log(4000))[0][-1]
        if idx_start_fit < idx_end_fit and idx_end_fit < len(scales):
             min_s_fit = scales[idx_start_fit]
             max_s_fit = scales[idx_end_fit]
             custom_scale_range_for_hq = (min_s_fit, max_s_fit)
             print(f"Using custom subrange of scales for fitting h(q): {custom_scale_range_for_hq}")
        else:
            custom_scale_range_for_hq = None
            print("Custom subrange cannot be defined. All valid scales will be used.")
    else:
        custom_scale_range_for_hq = None 
        print("There is no enough scales for defining a subrange. All valid scales will be used")
    
    # 3. Execute MFDFA
    q_valid, h, tau, alpha, f_alpha, Fqs = mfdfa(
        data=example_series,
        q_values=q_values,
        scales=scales,
        order=detrend_order, num_cores=num_cores_to_use,
        segments_from_both_ends=process_both, 
        scale_range_for_hq=custom_scale_range_for_hq   
    )
    
    # 4. Calculate crossovers using SPIC method
    q_to_analyze_idx = np.argmin(np.abs(q_valid - (-5.0))) # Analyze the case of q=-5.0
    q_to_analyze_val = q_valid[q_to_analyze_idx]
    # Obtain Fqs data for that q
    current_F_q_s_row_for_crossover = Fqs[q_to_analyze_idx, :]
    #Again apply valid mask for that q
    valid_F_mask_for_crossover = np.isfinite(current_F_q_s_row_for_crossover) & (current_F_q_s_row_for_crossover > 0)
    
    # Asegúrate de que haya suficientes puntos válidos para el crossover
    if np.sum(valid_F_mask_for_crossover) < 2: # O el mínimo que requiera find_best_crossovers
        print(f"No enought valid points for finding crossovers with q={q_to_analyze_val:.1f}")
    else:
        best_crossover_indices = SPIC(
            scales[valid_F_mask_for_crossover], 
            current_F_q_s_row_for_crossover[valid_F_mask_for_crossover],
            max_k_to_test=3, 
            num_permutations=50, 
            min_points_per_segment=3, 
            significance_level=0.05,
            n_jobs=4,
            use_numba=True
        )
        print(f"Crossover(s) found for q={q_to_analyze_val:.1f}: {best_crossover_indices}")
        
    # 4. Plot results
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MFDFA of a binomial cascade',fontsize=25)
    ax = axs[0, 0]
    ax.tick_params('both',labelsize=20)
    q_index_to_plot = [0, len(q_valid) // 4, len(q_valid) // 2 ,3*len(q_valid) // 4,len(q_valid) - 1]
    for i_q_idx in q_index_to_plot:
        q_val_current = q_valid[i_q_idx]
        if np.any(np.isfinite(Fqs[i_q_idx, :]) & (Fqs[i_q_idx, :] > 0)):
            valid_F_mask = np.isfinite(Fqs[i_q_idx, :]) & (Fqs[i_q_idx, :] > 0)
            ax.loglog(scales[valid_F_mask], Fqs[i_q_idx, valid_F_mask], 'o-', label=f'q={q_val_current:.1f}',zorder=1)

    ax.set_xlabel('Scale($s$)',fontsize=20)
    ax.set_ylabel('$F_q(s)$',fontsize=20)
    #.Custom subrange
    ax.axvspan(min_s_fit, max_s_fit, color='lightgray', alpha=0.5,zorder=0)
    ax.set_xlim(np.min(scales)*0.9,1.1*np.max(scales))
    ax.set_ylim(np.min(Fqs)*0.9,1.1*np.max(Fqs))
    legend = ax.legend(fontsize=15, frameon=True)
    legend.get_frame().set_facecolor('white')      
    legend.get_frame().set_alpha(0.7)             
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.0)
    ax.grid(True, which="both", ls="-", alpha=0.5)
    
    ax = axs[0, 1]
    ax.plot(q_valid, h, 'o-', color='darkblue')
    ax.set_xlabel('$q$',fontsize=20)
    ax.set_ylabel('$h(q)$',fontsize=20)
    ax.grid(True, alpha=0.5)
    ax.tick_params('both',labelsize=20)
    #.tau vs q
    ax = axs[1, 0]
    ax.plot(q_valid, tau, 'o-', color='darkblue')
    ax.set_xlabel('$q$',fontsize=20)
    ax.set_ylabel(r'$\tau(q)$',fontsize=20)
    ax.grid(True, alpha=0.5)
    ax.tick_params('both',labelsize=20)
    
    #.f(alpha) vs alpha
    ax = axs[1, 1]
    valid_spectrum_mask = ~np.isnan(alpha) & ~np.isnan(f_alpha)
    ax.plot(alpha[valid_spectrum_mask], f_alpha[valid_spectrum_mask], 'o-', color='darkblue')
    ax.set_xlabel(r'$\alpha$',fontsize=20)
    ax.set_ylabel(r'$D(\alpha)$',fontsize=20)
    ax.grid(True, alpha=0.5)
    ax.tick_params('both',labelsize=20)
    ax.text(0.95, 0.95, "(d)", transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

```
## Note on Monofractal Series

It is important to note that `mf-toolkit` is designed with rigorous validations. If MFDFA is applied to a series that is **monofractal** (or has a very weak multifractal structure), it is *expected* behavior for the library to return a very small range of valid `q`, or even none at all.

This is not an error, but a feature: the algorithm is correctly indicating that there is no rich, mathematically coherent multifractal spectrum in the signal. The signature of a monofractal series in this analysis is precisely an almost constant `h(q)` range and a very narrow singularity spectrum that does not pass the validation filters. In this case, please use DFA (Detrended Fluctuation Analysis).

## How to Cite

If you use `mf-toolkit` in your research, please cite it using the following DOI:

[![DOI](https://zenodo.org/badge/DOI/TU_DOI_AQUI.svg)](https://doi.org/TU_DOI_AQUI)


## License

This project is distributed under the MIT license. See the `LICENSE` file for more details.
