# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Base configuration management."""

import contextlib
import os
import typing


# -----------------------------------------------------------------------------
# --- Configuration paths
# -----------------------------------------------------------------------------

SUBFOLDER: typing.Final[str] = os.path.join('.anaconda', 'navigator')


def get_home_dir() -> str:
    """Return user home directory."""
    if os.name == 'nt':
        fallback: str = ''
        path_env_var: str
        path_env_vars: typing.Sequence[str] = ('HOME', 'APPDATA', 'USERPROFILE', 'TMP')

        # look through all available values
        for path_env_var in path_env_vars:
            current: str = os.environ.get(path_env_var, '')
            if (not current) or (not os.path.isdir(current)):
                continue

            # save first existing folder as a fallback value
            fallback = fallback or current

            # prefer options which already contain `.anaconda\navigator`
            inner: str = os.path.join(current, SUBFOLDER)
            if os.path.isdir(inner):
                return current

        # if there was any working option - use it
        if fallback:
            return fallback

    else:
        with contextlib.suppress(BaseException):
            return os.path.expanduser('~')

    raise RuntimeError('Please define environment variable $HOME')


def get_conf_path(filename: typing.Optional[str] = None) -> str:
    """Return absolute path for configuration file with specified filename."""
    result: str = os.path.join(get_home_dir(), SUBFOLDER)
    os.makedirs(result, exist_ok=True)

    if filename is not None:
        return os.path.join(result, filename)
    return result
