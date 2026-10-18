"""Manuscript research GHQ, separate from the released public Laplace API."""
from .api import run_ghq
from .profiles import get_profile, get_integration_plan

__version__ = '2026.10.18.2'
__all__ = ['run_ghq', 'get_profile', 'get_integration_plan']
