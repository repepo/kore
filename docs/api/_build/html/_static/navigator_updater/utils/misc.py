# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) Spyder Project Contributors
#
# Licensed under the terms of the MIT License
# (see spyder/__init__.py for details)
# -----------------------------------------------------------------------------

"""Miscellaneous utilities."""

import os
import os.path as osp
import shutil
import uuid
import psutil
from navigator_updater.config import LOCKFILE, PIDFILE


def __remove_pyc_pyo(fname):
    """Eventually remove .pyc and .pyo files associated to a Python script"""
    if osp.splitext(fname)[1] == '.py':
        for ending in ('c', 'o'):
            if osp.exists(fname + ending):
                os.remove(fname + ending)


def rename_file(source, dest):
    """
    Rename file from *source* to *dest*
    If file is a Python script, also rename .pyc and .pyo files if any
    """
    os.rename(source, dest)
    __remove_pyc_pyo(source)


def remove_file(fname):
    """
    Remove file *fname*
    If file is a Python script, also rename .pyc and .pyo files if any
    """
    os.remove(fname)
    __remove_pyc_pyo(fname)


def move_file(source, dest):
    """
    Move file from *source* to *dest*
    If file is a Python script, also rename .pyc and .pyo files if any
    """
    shutil.copy(source, dest)
    remove_file(source)


def abspardir(path):
    """Return absolute parent dir"""
    return osp.abspath(osp.join(path, os.pardir))


def get_common_path(pathlist):
    """Return common path for all paths in pathlist"""
    common = osp.normpath(osp.commonprefix(pathlist))
    if len(common) > 1:
        if not osp.isdir(common):
            return abspardir(common)
        for path in pathlist:
            if not osp.isdir(osp.join(common, path[len(common) + 1:])):
                # `common` is not the real common prefix
                return abspardir(common)
        return osp.abspath(common)
    return None


def path_is_writable(path):
    """Check if given path is writable."""
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(path)))

    path_exists = False
    if os.path.isfile(path):
        test_filepath = path
        remove = False
    else:
        path_exists = os.path.isdir(path)
        remove = True
        if not path_exists:
            try:
                os.makedirs(path)
            except Exception:  # pylint: disable=broad-except
                return False

        test_filepath = os.path.join(path, uuid.uuid4().hex[:5])

    try:
        fh = open(test_filepath, 'a+')  # pylint: disable=consider-using-with,invalid-name,unspecified-encoding
    except (IOError, OSError):
        return False

    fh.close()
    try:
        if remove:
            os.remove(test_filepath)
        if not path_exists:
            os.rmdir(path)
    except OSError:
        pass
    return True


def save_pid():
    """Save navigator process ID."""
    pid = os.getpid()

    try:
        with open(PIDFILE, 'w') as f:  # pylint: disable=invalid-name,unspecified-encoding
            f.write(str(pid))
    except Exception:  # pylint: disable=broad-except
        pid = None

    return pid


def load_pid():
    """Load navigator process ID."""
    try:
        with open(PIDFILE, 'r') as f:  # pylint: disable=invalid-name,unspecified-encoding
            pid = f.read()
        pid = int(pid)
    except Exception:  # pylint: disable=broad-except
        pid = None

    if pid is not None:
        is_running = psutil.pid_exists(pid)
        process = None
        cmds = []
        try:
            process = psutil.Process(pid)
            if process and is_running:
                cmds = process.cmdline()
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied:
            # Try to remove the pid file, if not possible return False
            if not remove_pid():
                return False

        cmds = [cmd.lower() for cmd in cmds]

        # Check bootstrap
        ch1 = [
            cmd
            for cmd in cmds
            if 'python' in cmd or 'bootstrap.py' in cmd
        ]
        ch2 = [
            cmd
            for cmd in cmds
            if 'python' in cmd or 'anaconda-navigator' in cmd
        ]
        ch3 = [
            cmd
            for cmd in cmds
            if 'navigator.app' in cmd
        ]

        check = any(ch1) or any(ch2) or any(ch3)

        if not check:
            pid = None

    return pid


def remove_pid():
    """Load navigator process ID."""
    check = True
    try:
        os.remove(PIDFILE)
    except Exception:  # pylint: disable=broad-except
        check = False
    return check


def remove_lock():
    """Load navigator process ID."""
    check = True
    try:
        os.remove(LOCKFILE)
    except Exception:  # pylint: disable=broad-except
        check = False
    return check


def set_windows_appusermodelid():
    """Make sure correct icon is used on Windows 7 taskbar"""
    try:
        from ctypes import windll  # pylint: disable=import-outside-toplevel
        name = 'anaconda.Anaconda-Navigator'
        return windll.shell32.SetCurrentProcessExplicitAppUserModelID(name)
    except AttributeError:
        return 'SetCurrentProcessExplicitAppUserModelID not found'


def convert_file_url_to_path(file_url):
    """Remove `file:///` from path."""
    return file_url.replace('file:///', '') if os.name == 'nt' else file_url.replace('file://', '')
