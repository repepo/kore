# -*- coding: utf-8 -*-

"""Additional utility functions to use with pycharm."""

from __future__ import annotations

__all__ = [
    'VSCodeVersionChecker', 'vscode_detector', 'vscode_extra_arguments', 'vscode_install_extensions',
    'vscode_update_config',
]

import datetime
import json
import os
import typing
from anaconda_navigator import config as navigator_config
from anaconda_navigator.api import conda_api
from anaconda_navigator.utils.conda import launch as conda_launch_utils
from anaconda_navigator.utils.logs import logger
from .. import detectors
from .. import validation_utils
from . import detector_utils

if typing.TYPE_CHECKING:
    from anaconda_navigator.api import process
    from .. import base
    from .. import parsing_utils


class VSCodeVersionChecker(detectors.Filter):  # pylint: disable=too-few-public-methods
    """Detect version of the VS Code application."""

    __slots__ = ()

    @classmethod
    def _parse_configuration(cls, *args: typing.Any, **kwargs: typing.Any) -> detectors.Detector:
        """Parse configuration for this particular :class:`~Detector`."""
        validation_utils.has_items(at_most=0)(args, field_name='args')

        validation_utils.mapping_is_empty()(kwargs)

        with validation_utils.catch_exception():
            return cls()

    def __call__(
            self,
            parent: typing.Iterator[detectors.DetectedApplication],
            *,
            context: detectors.DetectorContext,
    ) -> typing.Iterator[detectors.DetectedApplication]:
        """Iterate through detected applications."""
        application: 'detectors.DetectedApplication'
        for application in parent:
            if not application.executable:
                continue

            stdout: str
            stdout, _, _ = conda_launch_utils.run_process([application.executable, '--version'])
            if stdout:
                yield application.replace(version=stdout.splitlines()[0])
            else:
                yield application.replace(version='unknown')


def vscode_detector() -> detectors.Source:
    """Prepare detector for VS Code."""
    return detectors.Group(
        detectors.OsXOnly(
            detectors.Group(
                detectors.CheckConfiguredRoots(),
                detectors.CheckKnownOsXRoots('Visual Studio Code.app'),
                detectors.AppendExecutable(
                    detectors.join('Contents', 'Resources', 'app', 'bin', 'code'),
                ),
            ),

            detectors.Group(
                detectors.CheckPATH('code'),
                detectors.AppendRoot(level=4),
            ),
        ),

        detectors.LinuxOnly(
            detectors.Group(
                detectors.CheckConfiguredRoots(),
                detectors.CheckKnownRoots(
                    detectors.join(
                        detectors.FOLDERS['linux/root'], 'usr', 'share', 'code',
                    ),
                    detectors.join(
                        detectors.FOLDERS['linux/snap_primary'], 'code', 'current', 'usr', 'share', 'code',
                    ),
                    detectors.join(
                        detectors.FOLDERS['linux/snap_secondary'], 'code', 'current', 'usr', 'share', 'code',
                    ),
                ),
                detectors.AppendExecutable(
                    detectors.join('bin', 'code'),
                ),
            ),

            detectors.Group(
                detectors.CheckPATH('code'),
                detectors.AppendRoot(level=1),
            ),
        ),

        detectors.WindowsOnly(
            detectors.Group(
                detectors.CheckConfiguredRoots(),
                detectors.CheckKnownRoots(
                    detectors.join(detectors.FOLDERS['windows/program_files_x86'], 'Microsoft VS Code'),
                    detectors.join(detectors.FOLDERS['windows/program_files_x64'], 'Microsoft VS Code'),
                    detectors.join(detectors.FOLDERS['windows/local_app_data'], 'Programs', 'Microsoft VS Code'),
                ),
                detectors.AppendExecutable(
                    detectors.join('bin', 'code.cmd'),
                ),
            ),

            detectors.Group(
                detectors.CheckPATH('code.cmd'),
                detectors.AppendRoot(level=1),
            ),

            detectors.Group(
                detectors.CheckRootInWindowsRegistry(
                    detectors.RegistryKey(
                        root='HKEY_CURRENT_USER',
                        key=(
                            'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\'
                            '{771FD6B0-FA20-440A-A002-3B3BAC16DC50}_is1'
                        ),
                        sub_key='Inno Setup: App Path',
                    ),
                    detectors.RegistryKey(
                        root='HKEY_LOCAL_MACHINE',
                        key=(
                            'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\'
                            '{EA457B21-F73E-494C-ACAB-524FDE069978}_is1'
                        ),
                        sub_key='Inno Setup: App Path',
                    ),
                ),
                detectors.AppendExecutable(
                    detectors.join('bin', 'code.cmd'),
                ),
            ),

            detectors.Group(
                detectors.CheckExecutableInWindowsRegistry(
                    detectors.RegistryKey(
                        root='HKEY_CLASSES_ROOT',
                        key='vscode\\shell\\open\\command',
                        converter=detector_utils.extract_app_from_command,
                    ),
                ),
                detectors.AppendRoot(level=1),
            ),
        ),

        VSCodeVersionChecker(),
    )


def vscode_extra_arguments() -> typing.Sequence[str]:
    """Return default extra arguments for vscode."""
    return '--user-data-dir', os.path.join(navigator_config.CONF_PATH, 'Code')


def vscode_update_config(instance: 'base.BaseInstallableApp', prefix: str) -> None:  # pylint: disable=unused-argument
    """Update user config to use selected Python prefix interpreter."""
    try:
        _config_dir: str = os.path.join(navigator_config.CONF_PATH, 'Code', 'User')
        _config: str = os.path.join(_config_dir, 'settings.json')

        try:
            os.makedirs(_config_dir, exist_ok=True)
        except OSError as exception:
            logger.error(exception)
            return

        stream: typing.TextIO
        config_data: typing.Dict[str, typing.Any]
        if os.path.isfile(_config):
            try:
                with open(_config, 'rt', encoding='utf-8') as stream:
                    data = stream.read()
                vscode_create_config_backup(data)

                config_data = json.loads(data)
            except BaseException:  # pylint: disable=broad-except
                return
        else:
            config_data = {}

        pyexec: str = conda_api.get_pyexec(prefix)
        config_data.update({
            'python.experiments.optInto': ['pythonTerminalEnvVarActivation'],
            'python.terminal.activateEnvInCurrentTerminal': True,
            'python.terminal.activateEnvironment': True,
            'python.pythonPath': pyexec,
            'python.defaultInterpreterPath': pyexec,
            'python.condaPath': conda_api.get_pyscript(conda_api.CondaAPI().ROOT_PREFIX, 'conda'),
        })
        with open(_config, 'wt', encoding='utf-8') as stream:
            json.dump(config_data, stream, sort_keys=True, indent=4)

    except Exception as exception:  # pylint: disable=broad-except
        logger.error(exception)
        return


def vscode_create_config_backup(data: str) -> None:
    """
    Create a backup copy of the app configuration file `data`.

    Leave only the last 10 backups.
    """
    date: str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    _config_dir: str = os.path.join(navigator_config.CONF_PATH, 'Code', 'User')
    _config_bck: str = os.path.join(_config_dir, f'bck.{date}.navigator.settings.json')

    # Make the backup
    stream: typing.TextIO
    with open(_config_bck, 'wt', encoding='utf-8') as stream:
        stream.write(data)

    # Only keep the latest 10 backups
    files: typing.List[str] = [
        os.path.join(_config_dir, item)
        for item in os.listdir(_config_dir)
        if item.startswith('bck.') and item.endswith('.navigator.settings.json')
    ]
    path: str
    for path in sorted(files, reverse=True)[10:]:
        try:
            os.remove(path)
        except OSError:
            pass


def vscode_install_extensions(instance: 'base.BaseInstallableApp') -> 'process.ProcessWorker':
    """Install app extensions."""
    if instance.executable is None:
        return instance._process_api.create_process_worker(['python', '--version'])  # pylint: disable=protected-access

    cmd: typing.Sequence[str] = [
        instance.executable,
        '--install-extension',
        # 'ms-python.anaconda-extension-pack',
        # 'ms-python-anaconda-extension',
        'ms-python.python',
    ]
    return instance._process_api.create_process_worker(cmd)  # pylint: disable=protected-access
