"""Moved to ``clm.embed_backends`` (installed with the package); kept so old imports work."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from clm.embed_backends import OfflineBackend, Recipe, ServerBackend, make_backend  # noqa: E402,F401
