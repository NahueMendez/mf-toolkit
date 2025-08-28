from .MFDFA import mfdfa
from .crossovers import SPIC, CDVA
from .mfsources import generate_iaaft, shuffle_surrogate
from .mfgeneration import generate_fgn, generate_mf_corr, generate_mf_dist

__version__ = "1.0.0"

__all__ = [
    'mfdfa',
    'SPIC',
    'CDVA',
    'shuffle_surrogate',
    'generate_iaaft',
    'generate_fgn',
    'generate_mf_corr',
    'generate_mf_dist'
]
