""" Logging configs and extensions. """

import logging.config
import sys
import threading
import types
import typing

from navigator_updater.utils.logs.common import *
from navigator_updater.utils.logs.loggers import *
from navigator_updater.utils.logs.config import LOGGER_CONFIG


def global_exception_logger(
        exc_type: typing.Type[BaseException],
        exception: BaseException,
        traceback: typing.Optional[types.TracebackType],
) -> None:
    """Handle an exception by logs it."""
    logger.exception(exception, exc_info=(exc_type, exception, traceback))


def setup_logger() -> None:
    """Setup, create, and set logger."""
    logging.config.dictConfig(LOGGER_CONFIG.dict_config)

    # Note: threading.excepthook is only supported since Python 3.8
    sys.excepthook = global_exception_logger
    sys.unraisablehook = global_exception_logger  # type: ignore
    threading.excepthook = global_exception_logger  # type: ignore

    logger.debug('Setting up logger')
