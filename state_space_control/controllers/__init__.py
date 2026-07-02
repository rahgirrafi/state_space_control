"""Controller modules. Importing this package registers every available
design; controllers whose optional dependencies are missing are skipped."""

from . import lqg, lqr  # noqa: F401

try:
    from . import hinf  # noqa: F401  (needs python-control)
except ImportError:
    pass
