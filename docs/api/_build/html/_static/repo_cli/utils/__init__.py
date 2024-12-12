"""
Utilities functions for Anaconda repository command line manager
"""
from __future__ import print_function, unicode_literals

import logging
import sys
from logging.handlers import RotatingFileHandler
from os import getenv, makedirs, path
from os.path import exists, isfile, join

import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from six import PY2

from .config import USER_LOGDIR


def file_or_token(value):
    """
    If value is a file path and the file exists its contents are stripped and returned,
    otherwise value is returned.
    """
    if isfile(value):
        with open(value) as fd:
            return fd.read().strip()

    if any(char in value for char in "/\\."):
        # This chars will never be in a token value, but may be in a path
        # The error message will be handled by the parser
        raise ValueError()

    return value


def _custom_excepthook(logger, show_traceback=False):
    def excepthook(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            logger.error("execution interrupted")
            return

        if show_traceback:
            logger.error("", exc_info=(exc_type, exc_value, exc_traceback))
        else:
            logger.error("%s", exc_value)

    return excepthook


def get_ssl_verify_option(config, insecure, logger):
    """
    Tries to determine if SSL certificate verification is
    needed and if needed then the follows priorities below

    1. If `verify_ssl` is given true then it is highest priority,
    2. Then `SSL_NO_VERIFY` comes next if environment variable is set,
    3. Else if `ssl_verify` is set in config and if either string or boolean,
    4. Finally we should verify SSL certificates by default.
    """
    # Give priority to --insecure option if provided

    if insecure and insecure is not None:
        return False

    # Support environment setting like in conda
    insecure = getenv("SSL_NO_VERIFY")
    if insecure and insecure.isdigit() and int(insecure) == 1:
        logger.warning("SSL_NO_VERIFY is set please make sure to unset it.")
        return False

    # And least priority is given to setting based on configuration
    config_ssl_verify = config.get("ssl_verify")
    if isinstance(config_ssl_verify, (bool, str)):
        if isinstance(config_ssl_verify, bool) and not config_ssl_verify:
            logger.warning(
                '"ssl_verify" is set to false please consider to turn it on.'
            )

        if not path.exists(config_ssl_verify):
            logger.warning(
                'Specified "ssl_verify=%s" certificate path does not exist.'
                % config_ssl_verify
            )

        return config_ssl_verify

    return True


class ConsoleFormatter(logging.Formatter):
    def format(self, record):
        fmt = (
            "%(message)s"
            if record.levelno == logging.INFO
            else "[%(levelname)s] %(message)s"
        )
        if PY2:
            self._fmt = fmt
        else:
            self._style._fmt = fmt
        return super(ConsoleFormatter, self).format(record)


def _setup_logging(
    logger, log_level=logging.INFO, show_traceback=False, disable_ssl_warnings=False
):
    logger.setLevel(logging.DEBUG)

    if not exists(USER_LOGDIR):
        makedirs(USER_LOGDIR)

    log_file = join(USER_LOGDIR, "cli.log")

    file_handler = RotatingFileHandler(
        log_file, maxBytes=10 * (1024**2), backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(log_level)

    console_handler.setFormatter(ConsoleFormatter())
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)-15s %(message)s")
    )

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    sys.excepthook = _custom_excepthook(logger, show_traceback=show_traceback)

    if disable_ssl_warnings:
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)


def bool_input(prompt, default=True):
    default_str = "[Y|n]" if default else "[y|N]"
    while 1:
        inpt = input("%s %s: " % (prompt, default_str))
        if inpt.lower() in ["y", "yes"] and not default:
            return True
        elif inpt.lower() in ["", "n", "no"] and not default:
            return False
        elif inpt.lower() in ["", "y", "yes"]:
            return True
        elif inpt.lower() in ["n", "no"]:
            return False
        else:
            sys.stderr.write("please enter yes or no\n")
