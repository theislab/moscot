from importlib import metadata

import moscot.base
import moscot.costs
import moscot.utils
import moscot.backends
import moscot.datasets
import moscot.plotting
import moscot.problems

try:
    md = metadata.metadata(__name__)
    __version__ = md.get("version", "")
    __author__ = md.get("Author", "")
    __maintainer__ = md.get("Maintainer-email", "")
except ImportError:
    md = None

del metadata, md
