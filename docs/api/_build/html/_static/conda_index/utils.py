import filecmp
import hashlib
from concurrent.futures.thread import ThreadPoolExecutor
from contextlib import contextmanager

from conda.base.constants import (  # noqa: F401
    CONDA_PACKAGE_EXTENSION_V1,
    CONDA_PACKAGE_EXTENSION_V2,
    CONDA_PACKAGE_EXTENSIONS,
)
from conda.base.constants import PLATFORM_DIRECTORIES as DEFAULT_SUBDIRS

DEFAULT_SUBDIRS = set(DEFAULT_SUBDIRS)


def LoggingContext(*args, **kwargs):
    @contextmanager
    def log_context():
        """no-op logging context"""
        yield

    return log_context()


# multithreaded checksums


def _checksum(fd, algorithm, buffersize=65536):
    hash_impl = getattr(hashlib, algorithm)()
    for block in iter(lambda: fd.read(buffersize), b""):
        hash_impl.update(block)
    return hash_impl.hexdigest()


def checksum(fn, algorithm, buffersize=1 << 18):
    """
    Calculate a checksum for a filename (not an open file).
    """
    with open(fn, "rb") as fd:
        return _checksum(fd, algorithm, buffersize)


def checksums(fn, algorithms, buffersize=1 << 18):
    """
    Calculate multiple checksums for a filename in parallel.
    """
    with ThreadPoolExecutor(max_workers=len(algorithms)) as e:
        # take care not to share hash_impl between threads
        results = [
            e.submit(checksum, fn, algorithm, buffersize) for algorithm in algorithms
        ]
    return [result.result() for result in results]


from .utils_build import (  # noqa: F401
    ensure_list,
    get_lock,
    merge_or_update_dict,
    move_with_fallback,
    try_acquire_locks,
)


def file_contents_match(pathA, pathB):
    """
    Return True if pathA and pathB have identical contents.
    """

    return filecmp.cmp(pathA, pathB, shallow=False)
