from .MFDFA import mfdfa
from .crossovers import SPIC, CDVA
from .mfsources import iaaft_surrogate, shuffle_surrogate

__version__ = "0.1.2"

__all__ = [
    'mfdfa',
    'SPIC',
    'CDVA',
    'iaaft_surrogate',
    'shuffle_surrogate'
]
