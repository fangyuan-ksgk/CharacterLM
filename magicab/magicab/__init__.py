"""
Magicab is a character-level language model with variable vocabulary 
"""

__version__ = "0.1.0"
__author__ = "Fangyuan Yu"

from .magicab import Magicab, update_magicab, save_magicab
from .etoken import ETokenizer
from .vis import *
from .utils import *

# Optional: Define __all__ to control what gets imported with "from package import *"
__all__ = [
    'Magicab',
    'ETokenizer',
    # Add other important classes/functions you want to expose
]