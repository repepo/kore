# -*- coding: utf-8 -*-

"""Collection of the core utilities."""

from __future__ import annotations

__all__ = ['get_conda_cmd_path', 'is_conda_available', 'get_conda_info']

import itertools
import json
import os
import sys
import typing

from . import exceptions
from . import launch

if typing.TYPE_CHECKING:
    from . import types as conda_types


CONDA_PATH: str
if os.name == 'nt':
    CONDA_PATH = os.path.join('Scripts', 'conda-script.py')
else:
    CONDA_PATH = os.path.join('bin', 'conda')


def get_conda_cmd_path(*prefixes: str) -> typing.Optional[str]:
    """Check if conda is found on path."""
    commands: typing.List[str] = []

    conda_exe: str = os.environ.get('CONDA_EXE', '')
    if conda_exe:
        commands.append(conda_exe)

    prefix: str = os.path.abspath(sys.prefix)
    for prefix in itertools.chain(prefixes, [os.path.dirname(os.path.dirname(prefix)), prefix]):
        commands.append(os.path.join(prefix, CONDA_PATH))

    commands.append('conda')

    command: str
    for command in commands:
        stdout: str
        stderr: str
        error: bool
        stdout, stderr, error = launch.run_process(cmd_list=[command, '--version'])

        if (not error) and any(item.startswith('conda ') for item in (stdout, stderr)):
            return command

    return None


def is_conda_available() -> bool:
    """Check if conda is available in path."""
    return get_conda_cmd_path() is not None


def get_conda_info() -> typing.Optional['conda_types.CondaInfoOutput']:
    """Return conda info as a dictionary."""
    conda_cmd: typing.Optional[str] = get_conda_cmd_path()
    if conda_cmd is None:
        return None

    out: str = launch.run_process(cmd_list=[conda_cmd, 'info', '--json']).stdout

    result: typing.Union['conda_types.CondaInfoOutput', 'conda_types.CondaErrorOutput']
    try:
        result = json.loads(out)
    except (TypeError, ValueError):
        return None

    if 'error' not in result:
        return typing.cast('conda_types.CondaInfoOutput', result)

    raise exceptions.CondaError(error=typing.cast('conda_types.CondaErrorOutput', result))
