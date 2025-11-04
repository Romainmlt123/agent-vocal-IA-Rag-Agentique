"""
Agent Vocal Prof - Local Voice Tutoring System
"""

__version__ = "0.1.0"
__author__ = "Intelligence Lab"
__license__ = "MIT"

from .config import Config
from .utils import setup_logging, get_logger

__all__ = [
    "Config",
    "setup_logging",
    "get_logger",
]
