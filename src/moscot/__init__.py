from importlib import metadata

from moscot import backends, base, costs, datasets, plotting, problems, utils

try:
    md = metadata.metadata(__name__)
    __version__ = md.get("version", "")  # type: ignore[attr-defined]
    __author__ = md.get("Author", "")  # type: ignore[attr-defined]
    __maintainer__ = md.get("Maintainer-email", "")  # type: ignore[attr-defined]
except ImportError:
    md = None

del metadata, md
