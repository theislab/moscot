import moscot.costs
import moscot.solvers
import moscot.backends
import moscot.datasets
import moscot.plotting
import moscot.problems

__author__ = __maintainer__ = "Theislab"
__email__ = ", ".join(
    [
        "dominik.klein@helmholtz-muenchen",
        "michal.klein@protonmail.com",
        "giovanni.palla@helmholtz-muenchen.de",
    ]
)
__version__ = "0.1"

try:
    from importlib_metadata import version  # Python < 3.8
except ImportError:
    from importlib.metadata import version  # Python = 3.8

from packaging.version import parse

try:
    __full_version__ = parse(version(__name__))
    __full_version__ = f"{__version__}+{__full_version__.local}" if __full_version__.local else __version__
except ImportError:
    __full_version__ = __version__

del version, parse
