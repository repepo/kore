"""
conda-token: Configure Conda to utilize Anaconda Commercial Edition.
"""

from ._version import get_versions

__version__ = get_versions()['version']
del get_versions

from .repo_config import (ACTIVE_CHANNELS, ARCHIVE_CHANNELS,  # noqa
                          MAIN_CHANNEL, REPO_URL, token_list, token_remove,
                          token_set)
