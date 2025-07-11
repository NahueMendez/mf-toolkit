from .MFDFA import mfdfa
from .crossovers import SPIC, CDVA
from .mfsources import iaaft_surrogate, shuffle_surrogate

__version__ = "1.0.0"

__all__ = [
    'mfdfa',
    'SPIC',
    'CDVA',
    'iaaft_surrogate',
    'generate_iaaft'
]
