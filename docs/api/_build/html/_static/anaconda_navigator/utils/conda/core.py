# -*- coding: utf-8 -*-

"""Collection of the core utilities."""

from __future__ import annotations

import functools
import re

__all__ = ['get_conda_cmd_path', 'is_conda_available', 'get_conda_info']

import html
import itertools
import json
import os
import pathlib
import sys
import typing

import yaml

from anaconda_navigator.exceptions import CustomMessageException
from anaconda_navigator.utils import extra_collections
from anaconda_navigator.utils import notifications
from . import launch
from . import solvers

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
        stdout, stderr, error = launch.run_process(cmd_list=[command, '--version', '--json'])

        if error:
            continue

        if any(item.startswith('conda ') for item in (stdout, stderr)):
            return command

        if all(item == '' for item in (stdout, stderr)):  # if condarc is broken, conda may respond without any output
            return command

    return None


def is_conda_available() -> bool:
    """Check if conda is available in path."""
    return get_conda_cmd_path() is not None


class CondaRcManager:
    """Helper manager to validate and backup .condarc files."""

    __slots__ = ('__backups', '__broken', '__roots')

    ORIGINAL_NAME: typing.Final[str] = '.condarc'
    BACKUP_TEMPLATE: typing.Final[str] = '.condarc.backup_{index:d}'
    BACKUP_PATTERN: typing.Final[typing.Pattern[str]] = re.compile(r'^\.condarc\.backup_(\d+)$', re.M)

    def __init__(self) -> None:
        """Initialize new :class:`~CondaRcManager` instance."""
        self.__backups: typing.Final[typing.List[str]] = []
        self.__broken: typing.Final[typing.List[str]] = []
        self.__roots: typing.Optional[typing.Sequence[str]] = None

    @property
    def backups(self) -> typing.Sequence[str]:
        """
        Backups of original configuration files.

        Original file is usually removed if its backup is added to this list.
        """
        return self.__backups

    @property
    def broken(self) -> typing.Sequence[str]:
        """Files that have issues which Navigator is unable to fix."""
        return self.__broken

    @property
    def roots(self) -> typing.Sequence[str]:
        """Folders that may have configuration files within."""
        if self.__roots is not None:
            return self.__roots

        result: extra_collections.OrderedSet[str] = extra_collections.OrderedSet()
        result.add(os.path.expanduser('~'))

        option: typing.Optional[str] = os.environ.get('CONDA_PREFIX', None)
        if option:
            result.add(os.path.abspath(option))

        option = os.environ.get('CONDA_EXE', None)
        if option:
            result.add(str(pathlib.Path(option).parents[1].absolute()))

        try:
            stream: typing.TextIO
            with open(os.path.join(os.path.expanduser('~'), '.conda', 'environments.txt'), encoding='utf8') as stream:
                result.update(filter(bool, map(str.strip, stream)))
        except FileNotFoundError:
            pass

        self.__roots = tuple(result)
        return self.__roots

    def backup_all(self) -> None:
        """Backup all existing configuration files."""
        root: str
        for root in self.roots:
            self._create_backup(root)

    def notify(self) -> str:
        """
        Notify user about detected issues.

        This may end up with any kind of message box shown to the user.
        """
        message: str = self._prepare_message()
        if not message:
            return message
        if self.broken:
            raise CustomMessageException(message=message)
        if self.backups:
            notifications.NOTIFICATION_QUEUE.push(
                message=message,
                caption='Broken Conda configuration',
                tags=('conda', 'condarc'),
            )
        return message

    def validate(self) -> None:
        """Perform a quick validation of existing files."""
        root: str
        for root in self.roots:
            try:
                stream: typing.TextIO
                with open(os.path.join(root, self.ORIGINAL_NAME), 'r', encoding='utf8') as stream:
                    yaml.safe_load(stream)
            except FileNotFoundError:
                continue
            except yaml.YAMLError:
                self._create_backup(root)

    def _create_backup(self, root: str) -> None:
        """Backup configuration file located in `root` folder."""
        index: int = 1
        try:
            index += functools.reduce(max, map(int, self.BACKUP_PATTERN.findall('\n'.join(os.listdir(root)))), 0)
        except OSError:
            pass

        source: str = os.path.join(root, self.ORIGINAL_NAME)
        target: str = os.path.join(root, self.BACKUP_TEMPLATE.format(index=index))

        try:
            os.rename(source, target)
        except FileNotFoundError:
            pass
        except OSError:
            self.__broken.append(source)
        else:
            self.__backups.append(target)

    def _prepare_message(self) -> str:
        """
        Compile a message with details on affected configuration files.

        If nothing is affected - empty message is returned.
        """
        result: str = ''

        if self.broken:
            result += (
                f'<h1>There is a problem with your {self.ORIGINAL_NAME} files</h1>'
                f'<p>Unable to start Anaconda Navigator because of broken {self.ORIGINAL_NAME} files.</p>'
            )
        elif self.backups:
            result = f'<h1>There was a problem with your {self.ORIGINAL_NAME} files</h1>'

        value: str
        if self.backups:
            result += (
                '<hr><p>'
                f'Some {self.ORIGINAL_NAME} files were reset automatically. '
                'You can find their original content in respective backup files:'
            )
            for value in self.backups:
                result += f'<br>- {html.escape(value)}'
            result += '</p>'

        if self.broken:
            result += (
                '<hr><p>'
                f'Navigator was unable to fix next {self.ORIGINAL_NAME} files. '
                'Please check their permissions and content:'
            )
            for value in self.broken:
                result += f'<br>- {html.escape(value)}'
            result += '</p>'

        return result


def get_conda_info(allow_fixes: bool = False) -> typing.Optional['conda_types.CondaInfoOutput']:
    """Return conda info as a dictionary."""
    conda_cmd: typing.Optional[str] = get_conda_cmd_path()
    if conda_cmd is None:
        return None

    manager: CondaRcManager = CondaRcManager()
    manager.validate()
    manager.notify()

    while True:
        out: str = launch.run_process(cmd_list=[conda_cmd, 'info', '--json']).stdout

        result: typing.Union['conda_types.CondaInfoOutput', 'conda_types.CondaErrorOutput']
        try:
            result = json.loads(out)
        except (TypeError, ValueError):
            return None

        if 'error' not in result:
            return typing.cast('conda_types.CondaInfoOutput', result)

        error: 'conda_types.CondaErrorOutput' = typing.cast('conda_types.CondaErrorOutput', result)
        if allow_fixes and (solvers.POOL.solve(error=error) is not None):
            continue

        manager.backup_all()
        if not manager.notify():
            message: str = (
                '<h1>Unable to start Anaconda Navigator</h1>'
                '<p>Navigator is unable to contact conda.</p>'
                '<p>Run '
                '<span style="border: 1px solid grey; border-radius: 2px; font-family: monospace">conda info</span> '
                'to get more details on the issue.</p>'
            )
            raise CustomMessageException(message=message)
