"""
Smart AI Agent Package
"""

from .core import SmartAgent
from .vector_db import VectorDBManager
from .web_search import WebSearchManager
from .tools import AgentTools
from .config import Config

__version__ = "1.0.0"
__all__ = ["SmartAgent", "VectorDBManager", "WebSearchManager", "AgentTools", "Config"]
