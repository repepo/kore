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
import shutil
import sys
import typing


# -----------------------------------------------------------------------------
# --- Configuration paths
# -----------------------------------------------------------------------------

CONFIG_NAME: typing.Final[str] = 'anaconda-navigator'
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


def is_ubuntu() -> bool:
    """Detect if we are running in an Ubuntu-based distribution."""
    if sys.platform.startswith('linux'):
        try:
            stream: typing.TextIO
            with open('/etc/lsb-release', 'rt', encoding='utf-8') as stream:
                return 'ubuntu' in stream.read().lower()
        except OSError:
            pass
    return False


def is_gtk_desktop() -> bool:
    """Detect if we are running in a Gtk-based desktop."""
    if sys.platform.startswith('linux'):
        return any(map(os.environ.get('XDG_CURRENT_DESKTOP', '').startswith, ['Unity', 'GNOME', 'XFCE']))
    return False  # type: ignore


def fix_recursive_folder() -> None:
    """Check and fix `.anaconda/navigator/.anaconda/navigator` folder issue."""
    root: str = get_conf_path()
    file: str = f'{CONFIG_NAME}.ini'

    # check if expected configuration file already exists
    if os.path.isfile(os.path.join(root, file)):
        return

    # check if invalid configuration file already exists
    inner: str = os.path.join(root, SUBFOLDER)
    if not os.path.isfile(os.path.join(inner, file)):
        return

    # move old configurations to the new location
    name: str
    for name in os.listdir(inner):
        try:
            os.rename(os.path.join(inner, name), os.path.join(root, name))
        except OSError:
            pass

    # create a link, so old location shares the content with a new location
    try:
        inner = os.path.dirname(inner)
        root = os.path.dirname(root)
        shutil.rmtree(inner)
        os.symlink(root, inner)
    except OSError:
        pass
