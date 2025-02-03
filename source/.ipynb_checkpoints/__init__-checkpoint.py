__all__ = ["config",
           "propagator",
           "optical_mul"]
__version__ = "2.0.1"

from .config import Config
from . import propagator
from .optical_mul import OpticalMul
from .parallel import DataParallel