"""
AstroAgent Pipeline - Services Package

This package contains service modules for external data access and analysis:
- search_ads: ADS and arXiv literature search
- literature: Literature analysis and novelty checking
- datasets: Astronomical survey data access
- ephemeris: Solar system ephemerides and orbits
- analysis: Statistical analysis utilities
- viz: Visualization utilities  
- provenance: Data provenance and hashing
"""

from .search_ads import ADSSearchService
from .literature import LiteratureService

__all__ = [
    'ADSSearchService',
    'LiteratureService',
]
