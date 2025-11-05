"""
Agent Vocal Prof - Local Voice Tutoring System with Pipecat
"""

__version__ = "0.2.0"
__author__ = "Intelligence Lab"
__license__ = "MIT"

# Imports conditionnels pour compatibilité
try:
    from .legacy.config import Config
    from .legacy.utils import setup_logging, get_logger
    __all__ = ["Config", "setup_logging", "get_logger"]
except ImportError:
    # Si legacy n'est pas disponible, ne rien exporter
    __all__ = []
