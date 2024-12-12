# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Launch applications utilities."""

import datetime
import itertools
import os
import subprocess  # nosec
import sys
import typing
from navigator_updater.config import LAUNCH_SCRIPTS_PATH, WIN


if WIN:
    import ctypes


DEVNULL: typing.Final[str] = 'nul' if (sys.platform == 'win32') else '/dev/null'
LOG_FILE: typing.Final[str] = '{package}-{level}-{index}.txt'


class CommandDetails(typing.TypedDict):
    """Dictionary with extra details about executed command."""

    args: str
    id: int  # pylint: disable=invalid-name
    cmd: str
    stdout: str
    stderr: str


class GetCommand(typing.Protocol):  # pylint: disable=too-few-public-methods
    """Common interface for all `get_command_*` functions."""

    def __call__(  # pylint: disable=too-many-arguments
            self,
            root_prefix: str,
            prefix: str,
            command: str,
            extra_arguments: typing.Iterable[typing.Any] = (),
            package_name: str = 'app',
            environment: typing.Optional[typing.Mapping[str, str]] = None,
            cwd: str = os.path.expanduser('~'),
            default_scripts_path: str = LAUNCH_SCRIPTS_PATH,
    ) -> typing.Tuple[typing.Mapping[str, typing.Any], CommandDetails]:
        """
        Generate script to launch application and return path to it.

        :param root_prefix: Path to root (base) conda prefix. Used to activate `prefix`.
        :param prefix: Conda prefix, which should be active.
        :param command: Actual command to launch application.
        :param extra_arguments: Additional arguments to attach to command.
        :param cwd: Custom working directory to launch application in.
        :param package_name: Name of the conda package, or alias of the external application.
        :param environment: Custom environment to launch application in.
        :param default_scripts_path: Root directory to store launch scripts in.
        :return: Path to generated launch script file.
        """


def safe_unix(value: str) -> str:
    """Prepare argument which is safe to use in unix command line."""
    to_replace: typing.Final[typing.Set[str]] = {'"', '$', '\\', '`'}
    to_escape: typing.Final[typing.Set[str]] = {' ', '&', ';', '<', '>', '|'} | to_replace

    if set(value) & to_escape:
        character: str
        for character in to_replace:
            value = value.replace(character, f'\\{character}')

        value = f'"{value}"'

    return value


def safe_windows(value: str) -> str:
    """Prepare argument which is safe to use in windows command line."""
    to_escape: typing.Final[typing.Set[str]] = {' ', '%', '"', '&', ';', '<', '>', '|'}

    if set(value) & to_escape:
        value = value.replace('"', '""').replace('%', '"^%"')
        value = f'"{value}"'

    return value


def expand_environment(
        command: str,
        environment: typing.Mapping[str, typing.Any],
        default: typing.Any = None,
        recursive: bool = False,
) -> str:
    """
    Update command with environment variables.

    :param command: Original command to update with environment variable values.
    :param environment: Environment variables to inject into command.

                        :code:`None` values won't be expanded.
    :param default: Value to insert for all unknown variables in `command`.
    :param recursive: Allow recursive expand of the values.
    :return: String with expanded environment.
    """
    cursor: int = 0
    while True:
        start: int = command.find('${', cursor)
        if start < 0:
            break

        stop: int = command.find('}', start)
        environment_key: str = command[start + 2:stop]
        stop += 1

        environment_value: typing.Any = environment.get(environment_key, default)
        if environment_value is None:
            cursor = stop
            continue

        environment_value = str(environment_value)
        command = command[:start] + environment_value + command[stop:]
        cursor = start
        if not recursive:
            cursor += len(environment_value)

    return command


class RunningProcess:
    """
    Minimal description of a process launched from the Navigator.

    Common example of such process - anything started from the Home page tiles.

    :param package: Name of the Conda package, or alias of a launched application.
    :param process: :class:`~subprocess.Popen` instance, which is used to launch the application.
    :param stdout: Path to a file with captured `stdout`.
    :param stderr: Path to a file with captured `stderr`.
    """

    __slots__ = ('__birth', '__package', '__process', '__return_code', '__stderr', '__stdout')

    def __init__(
            self,
            package: str,
            process: subprocess.Popen,
            stdout: typing.Optional[str] = None,
            stderr: typing.Optional[str] = None,
    ) -> None:
        """Initialize new :class:`~RunningProcess` instance."""
        if stdout == DEVNULL:
            stdout = None
        if stderr == DEVNULL:
            stderr = None

        self.__package: typing.Final[str] = package
        self.__process: typing.Final[subprocess.Popen] = process
        self.__return_code: typing.Optional[int] = None
        self.__stdout: typing.Final[typing.Optional[str]] = stdout
        self.__stderr: typing.Final[typing.Optional[str]] = stderr
        self.__birth: typing.Final[datetime.datetime] = datetime.datetime.utcnow()

    @property
    def age(self) -> datetime.timedelta:
        """Current age of the application."""
        return datetime.datetime.utcnow() - self.__birth

    @property
    def package(self) -> str:  # noqa: D401
        """Name of the Conda package, or alias of a launched application."""
        return self.__package

    @property
    def pid(self) -> int:  # noqa: D401
        """PID of the process."""
        return self.__process.pid

    @property
    def return_code(self) -> typing.Optional[int]:  # noqa: D401
        """Current status of the application."""
        if self.__return_code is None:
            self.__return_code = self.__process.poll()
        return self.__return_code

    @property
    def stderr(self) -> typing.Optional[str]:  # noqa: D401
        """Content of the `stdout` log."""
        return self.__read(self.__stderr)

    @property
    def stdout(self) -> typing.Optional[str]:  # noqa: D401
        """Content of the `stdout` log."""
        return self.__read(self.__stdout)

    @staticmethod
    def __read(path: typing.Optional[str]) -> typing.Optional[str]:
        """Read content of the file."""
        if path is None:
            return None
        try:
            stream: typing.TextIO
            with open(path, 'rt') as stream:  # pylint: disable=unspecified-encoding
                return stream.read()
        except OSError:
            return None

    def cleanup(self) -> None:
        """Remove all temporary content."""
        path: str
        for path in typing.cast(typing.Iterable[str], filter(bool, [self.__stdout, self.__stderr])):
            try:
                os.remove(path)
            except OSError:
                pass


def get_scripts_path(
        root_prefix: str,
        prefix: str,
        default_scripts_path: str = LAUNCH_SCRIPTS_PATH,
) -> str:
    """
    Return path to directory, where all launch scripts should be placed.

    This directory can also be used for the application logs.

    :param root_prefix: Path to root (base) conda prefix. Used to activate `prefix`.
    :param prefix: Conda prefix, which should be active.
    :param default_scripts_path: Root directory to store launch scripts in.
    :return: Path to directory, where launch script should be placed.
    """
    result: str = os.path.abspath(default_scripts_path)

    root_prefix = os.path.abspath(root_prefix)
    prefix = os.path.abspath(prefix)
    if prefix != root_prefix:
        result = os.path.join(result, os.path.basename(prefix))

    os.makedirs(result, exist_ok=True)
    return result


def remove_package_logs(
        root_prefix: str,
        prefix: str,
        default_scripts_path: str = LAUNCH_SCRIPTS_PATH,
) -> None:
    """
    Try to remove output, error logs for launched applications.

    :param root_prefix: Path to root (base) conda prefix. Used to activate `prefix`.
    :param prefix: Conda prefix, which should be active.
    :param default_scripts_path: Root directory to store launch scripts in.
    """
    logs_root: str = get_scripts_path(
        root_prefix=root_prefix,
        prefix=prefix,
        default_scripts_path=default_scripts_path,
    )

    file_path: str
    for file_path in os.listdir(logs_root):
        if not file_path.endswith('.txt'):
            continue
        try:
            os.remove(os.path.join(logs_root, file_path))
        except OSError:
            pass


def get_package_logs(
    root_prefix: str,
    prefix: str,
    package_name: str = 'app',
    id_: typing.Optional[int] = None,
    default_scripts_path: str = LAUNCH_SCRIPTS_PATH,
) -> typing.Tuple[str, str, int]:
    """
    Return the package log names for launched applications.

    :param root_prefix: Path to root (base) conda prefix. Used to activate `prefix`.
    :param prefix: Conda prefix, which should be active.
    :param package_name: Name of the conda package, or alias of the external application.
    :param id_: Application session identifier.

                Used to create different names for different application launch sessions.
    :param default_scripts_path: Root directory to store launch scripts in.
    """
    logs_root: str = get_scripts_path(
        root_prefix=root_prefix,
        prefix=prefix,
        default_scripts_path=default_scripts_path,
    )

    stdout_log_path: str = LOG_FILE.format(package=package_name, level='out', index=id_)
    stderr_log_path: str = LOG_FILE.format(package=package_name, level='err', index=id_)
    if id_ is None:
        used: typing.Set[str] = set(os.listdir(logs_root))

        for id_ in itertools.count(start=1):  # pylint: disable=redefined-argument-from-local
            stdout_log_path = LOG_FILE.format(package=package_name, level='out', index=id_)
            stderr_log_path = LOG_FILE.format(package=package_name, level='err', index=id_)
            if (stdout_log_path not in used) and (stderr_log_path not in used):
                break

    if prefix and root_prefix:
        stdout_log_path = os.path.join(logs_root, stdout_log_path)
        stderr_log_path = os.path.join(logs_root, stderr_log_path)

    return stdout_log_path, stderr_log_path, id_


def create_app_run_script(  # pylint: disable=too-many-arguments
    root_prefix: str,
    prefix: str,
    command: str,
    extension: str,
    package_name: str = 'app',
    default_scripts_path: str = LAUNCH_SCRIPTS_PATH,
) -> str:
    """
    Create new application launching script file.

    :param root_prefix: Path to root (base) conda prefix. Used to activate `prefix`.
    :param prefix: Conda prefix, which should be active.
    :param command: Content of the script file.
    :param extension: Extension for the script file (`.bat`, `.sh`)
    :param package_name: Name of the conda package, or alias of the external application.
    :param default_scripts_path: Root directory to store launch scripts in.
    :return: Path to generated launch script file.
    """
    scripts_root: str = get_scripts_path(
        root_prefix=root_prefix,
        prefix=prefix,
        default_scripts_path=default_scripts_path,
    )

    encoding: str = 'utf-8'
    if WIN:
        encoding = f'cp{ctypes.cdll.kernel32.GetACP()}'

    stream: typing.TextIO
    file_path: str = os.path.join(scripts_root, f'{package_name}{extension}')
    with open(file_path, 'wt', encoding=encoding) as stream:
        stream.write(command)

    os.chmod(file_path, 0o755)  # nosec

    return file_path


def get_command_on_win(  # pylint: disable=too-many-arguments
        root_prefix: str,
        prefix: str,
        command: str,
        extra_arguments: typing.Iterable[typing.Any] = (),
        package_name: str = 'app',
        environment: typing.Optional[typing.Mapping[str, str]] = None,
        cwd: str = os.path.expanduser('~'),
        default_scripts_path: str = LAUNCH_SCRIPTS_PATH,
) -> typing.Tuple[typing.Mapping[str, typing.Any], 'CommandDetails']:
    """
    Generate script to launch application and return path to it.

    This function is optimized to run on Windows.

    :param root_prefix: Path to root (base) conda prefix. Used to activate `prefix`.
    :param prefix: Conda prefix, which should be active.
    :param command: Actual command to launch application.
    :param extra_arguments: Additional arguments to attach to command.
    :param cwd: Custom working directory to launch application in.
    :param package_name: Name of the conda package, or alias of the external application.
    :param environment: Custom environment to launch application in.
    :param default_scripts_path: Root directory to store launch scripts in.
    :return: Path to generated launch script file.
    """
    stdout_log_path: str
    stderr_log_path: str
    id_: int
    stdout_log_path, stderr_log_path, id_ = get_package_logs(
        root_prefix=root_prefix,
        prefix=prefix,
        package_name=package_name,
        default_scripts_path=default_scripts_path,
    )

    command = dict((
        (
            'start cmd.exe /K "${CONDA_ROOT_PREFIX}\\\\Scripts\\\\activate.bat" "${CONDA_PREFIX}"',
            'start cmd.exe \\K "${CONDA_ROOT_PREFIX}\\Scripts\\activate.bat" "${CONDA_PREFIX}"',
        ),
        (
            str(
                'start powershell.exe -ExecutionPolicy ByPass -NoExit -Command  '  # pylint: disable=implicit-str-concat
                '"& \'{CONDA_ROOT_PREFIX}\\\\shell\\\\condabin\\\\conda-hook.ps1\' ; conda activate \'{CONDA_PREFIX}\'"'
            ),
            str(
                'start powershell.exe -ExecutionPolicy ByPass -NoExit -Command '  # pylint: disable=implicit-str-concat
                '"& \'${CONDA_ROOT_PREFIX}\\shell\\condabin\\conda-hook.ps1\' ; conda activate \'${CONDA_PREFIX}\'"'
            ),
        )
    )).get(command, command)

    command = expand_environment(
        command=command,
        environment={
            'PREFIX': prefix,
            'CONDA_PREFIX': prefix,
            'CONDA_ROOT_PREFIX': root_prefix,
        },
    )
    activate: str = os.path.join(root_prefix, 'Scripts', 'activate')

    extra_argument: typing.Any
    for extra_argument in extra_arguments:
        command += f' {safe_windows(str(extra_argument))}'

    script: str = '\n'.join([
        # disable echoing commands to the command line
        '@echo off',

        f'chcp {ctypes.cdll.kernel32.GetACP()}',

        # "call" is needed to avoid the batch script from closing after running the first (environment activation) line
        f'call {safe_windows(activate)} {safe_windows(prefix)}',

        f'{command} >{safe_windows(stdout_log_path)} 2>{safe_windows(stderr_log_path)}',

        '',
    ])

    file_path: str = create_app_run_script(
        root_prefix=root_prefix,
        prefix=prefix,
        command=script,
        extension='.bat',
        package_name=package_name,
        default_scripts_path=default_scripts_path,
    )
    return {  # popen_dict
        'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,  # type: ignore
        'cwd': cwd,
        'env': environment,
    }, {
        'args': file_path,
        'id': id_,
        'cmd': script,
        'stdout': stdout_log_path,
        'stderr': stderr_log_path,
    }


def get_command_on_unix(  # pylint: disable=too-many-arguments
        root_prefix: str,
        prefix: str,
        command: str,
        extra_arguments: typing.Iterable[typing.Any] = (),
        package_name: str = 'app',
        environment: typing.Optional[typing.Mapping[str, str]] = None,
        cwd: str = os.path.expanduser('~'),
        default_scripts_path: str = LAUNCH_SCRIPTS_PATH,
) -> typing.Tuple[typing.Mapping[str, typing.Any], 'CommandDetails']:
    """
    Generate script to launch application and return path to it.

    This function is optimized to run on Linux/OS X.

    :param root_prefix: Path to root (base) conda prefix. Used to activate `prefix`.
    :param prefix: Conda prefix, which should be active.
    :param command: Actual command to launch application.
    :param extra_arguments: Additional arguments to attach to command.
    :param cwd: Custom working directory to launch application in.
    :param package_name: Name of the conda package, or alias of the external application.
    :param environment: Custom environment to launch application in.
    :param default_scripts_path: Root directory to store launch scripts in.
    :return: Path to generated launch script file.
    """
    stdout_log_path: str
    stderr_log_path: str
    id_: int
    stdout_log_path, stderr_log_path, id_ = get_package_logs(
        root_prefix=root_prefix,
        prefix=prefix,
        package_name=package_name,
        default_scripts_path=default_scripts_path,
    )

    command = expand_environment(
        command=command,
        environment={
            'PREFIX': prefix,
            'CONDA_PREFIX': prefix,
            'CONDA_ROOT_PREFIX': root_prefix,
        },
    )
    activate: str = os.path.join(root_prefix, 'bin', 'activate')

    extra_argument: typing.Any
    for extra_argument in extra_arguments:
        command += f' {safe_unix(str(extra_argument))}'

    script: str = '\n'.join([
        '#!/usr/bin/env bash',
        f'. {safe_unix(activate)} {safe_unix(prefix)}',
        f'{command} >{safe_unix(stdout_log_path)} 2>{safe_unix(stderr_log_path)}',
        ''
    ])

    file_path: str = create_app_run_script(
        root_prefix=root_prefix,
        prefix=prefix,
        command=script,
        extension='.sh',
        package_name=package_name,
        default_scripts_path=default_scripts_path
    )
    return {
        'cwd': cwd,
        'env': environment,
    }, {
        'args': file_path,
        'id': id_,
        'cmd': script,
        'stdout': stdout_log_path,
        'stderr': stderr_log_path,
    }


def launch(  # pylint: disable=too-many-arguments
        root_prefix: str,
        prefix: str,
        command: str,
        extra_arguments: typing.Iterable[typing.Any] = (),
        working_directory: str = os.path.expanduser('~'),
        package_name: str = 'app',
        environment: typing.Optional[typing.Mapping[str, str]] = None,
        leave_path_alone: bool = True,  # pylint: disable=unused-argument
        non_conda: bool = False,  # pylint: disable=unused-argument
) -> typing.Optional[RunningProcess]:
    """
    Handle launching commands from projects.

    :param root_prefix: Path to root (base) conda prefix. Used to activate `prefix`.
    :param prefix: Conda prefix, which should be active.
    :param command: Actual command to launch application.
    :param extra_arguments: Additional arguments to attach to command.
    :param working_directory: Custom working directory to launch application in.

                              If not provided - home directory will be used.
    :param package_name: Name of the conda package, or alias of the external application.
    :param environment: Custom environment to launch application in.
    :param as_admin: Launch application with admin rights.

                     This breaks function result (changes it to :code:`True`/:code:`None`).
    :return: Description of the launched process.
    """
    if isinstance(extra_arguments, str):
        extra_arguments = (extra_arguments,)

    get_command: 'GetCommand' = get_command_on_unix
    if WIN:
        get_command = get_command_on_win

    popen_dict: typing.Mapping[str, typing.Any]
    extra_args: 'CommandDetails'
    popen_dict, extra_args = get_command(
        root_prefix=root_prefix,
        prefix=prefix,
        command=command,
        extra_arguments=extra_arguments,
        package_name=package_name,
        environment=environment,
        cwd=working_directory,
    )

    args: str = extra_args['args']
    if WIN:
        args = f'cmd.exe /C {safe_windows(args)}'

    return RunningProcess(
        package=package_name,
        process=subprocess.Popen(args, **popen_dict),  # nosec
        stdout=extra_args['stdout'],
        stderr=extra_args['stderr'],
    )
