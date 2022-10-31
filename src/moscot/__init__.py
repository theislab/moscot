from importlib import metadata

import moscot.costs
import moscot.solvers
import moscot.backends
import moscot.datasets
import moscot.plotting
import moscot.problems

try:
    __version__ = metadata.version(__name__)
    md = metadata.metadata(__name__)
    __author__ = md.get("Author", "")
    __maintainer__ = md.get("Maintainer-email", "")
except ImportError:
    md = None

del metadata, md
