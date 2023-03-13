from importlib import metadata

from moscot import backends, base, costs, datasets, plotting, problems, utils

try:
    md = metadata.metadata(__name__)
    __version__ = md.get("version", "")
    __author__ = md.get("Author", "")
    __maintainer__ = md.get("Maintainer-email", "")
except ImportError:
    md = None

del metadata, md
