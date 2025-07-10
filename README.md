# mf-toolkit

[![PyPI version](https://badge.fury.io/py/mf-toolkit.svg)](https://badge.fury.io/py/mf-toolkit)
[![DOI](https://zenodo.org/badge/DOI/TU_DOI_AQUI.svg)](https://doi.org/TU_DOI_AQUI)

A high-performance Python toolkit for the analysis of multifractal time series and complex systems.

This project provides robust and efficient implementations of key algorithms such as MFDFA, as well as methods for crossover detection and surrogate data generation for hypothesis testing.

## Main Features

* Fast MFDFA:** Efficient implementation of MFDFA analysis.
* Parallel Processing:** Optimized to utilize multiple CPU cores.
* Dual Engine:** Uses **Numba** for maximum performance or a Scikit-learn backend for easy installation.
* Crossover Detection:** Includes SPIC and CDVA methods to identify crossover changes.
* Surrogate Generation:** Creates test data with IAAFT and Shuffling methods.
* Robust Validation:** The code includes validations to ensure scientifically consistent results.

## Installation

You can install the stable version from PyPI:

```bash
pip install mf-toolkit
```

Para instalar la versión con el rendimiento optimizado por Numba, usa:
```bash
pip install mf-toolkit[numba]
```

## Quick Use

Here is a simple example of how to use the main `mfdfa` function:

```python
import numpy as np
import mftoolkit

# 1. Genera una serie de datos de ejemplo (ej. ruido blanco)
serie_de_ejemplo = np.random.randn(8192)

# 2. Define los parámetros para el análisis
q_valores = np.arange(-5, 5.1, 0.5)
escalas = np.logspace(np.log10(16), np.log10(1024), 20).astype(int)

# 3. Ejecuta el MFDFA
q, h, tau, alpha, f_alpha, fqs = mftoolkit.mfdfa(
    data=serie_de_ejemplo,
    q_values=q_valores,
    scales=escalas
)

# 4. Imprime el exponente de Hurst para q=2
print(f"Exponente de Hurst (h(q=2)): {h[q == 2][0]:.3f}")

```

## How to Cite

If you use `mf-toolkit` in your research, please cite it using the following DOI:

[![DOI](https://zenodo.org/badge/DOI/TU_DOI_AQUI.svg)](https://doi.org/TU_DOI_AQUI)


## License

This project is distributed under the MIT license. See the `LICENSE` file for more details.
