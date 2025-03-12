"""
Magicab is a character-level language model with variable vocabulary 
"""

__version__ = "0.1.0"
__author__ = "Fangyuan Yu"

from .magicab import Magicab, update_magicab, save_magicab, evaluate_token_stat
from .etoken import ETokenizer
from .data import get_batch_slice, save_sequences_for_memmap
from .vis import *
from .utils import *
from .data import *

# Optional: Define __all__ to control what gets imported with "from package import *"
__all__ = [
    'Magicab',
    'ETokenizer',
    # Add other important classes/functions you want to expose
]