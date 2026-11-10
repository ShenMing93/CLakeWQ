"""Spectral-only OLCI benchmark with versioned, reusable fitted models."""
__version__ = "0.1.0"

# Optional project-local dependencies; the global Python environment is untouched.
from pathlib import Path
import sys
_vendor = Path(__file__).resolve().parents[1] / '.vendor'
if _vendor.is_dir():
    sys.path.insert(0, str(_vendor))

