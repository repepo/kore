"""anaconda.catalogs is a library allowing users to interface with Anaconda's Intake catalogs service."""
import importlib.metadata
from typing import Optional

from intake import Catalog

# __version__ is imported before the catalog subclass
# to avoid circular imports
__version__ = importlib.metadata.version("anaconda-catalogs")

from .catalog import AnacondaCatalog

__all__ = ["__version__", "open_catalog", "AnacondaCatalog"]


def open_catalog(name: str, base_uri: Optional[str] = None) -> Catalog:
    """Open an Intake catalog hosted on Anaconda Catalogs."""
    return AnacondaCatalog(name=name, base_uri=base_uri)
