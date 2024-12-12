# -*- coding: utf-8 -*-

# pylint: disable=invalid-name,missing-function-docstring,too-many-arguments,too-many-lines,unspecified-encoding
# pylint: disable=unused-argument

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""
Updated `conda-api` running on a Qt QProcess to avoid UI blocking.

This also add some extra methods to the original conda-api.
"""

import contextlib
import hashlib
import itertools
import json
import os
import platform
import re
import subprocess  # nosec
import sys
import typing
from collections import deque

import yaml
from qtpy.QtCore import QByteArray, QObject, QProcess, QTimer, Signal

from navigator_updater.api import types as api_types
from navigator_updater.api.utils import split_canonical_name
from navigator_updater.config import WIN, get_home_dir, CONF_PATH
from navigator_updater.utils import ansi_utlils
from navigator_updater.utils import misc
from navigator_updater.utils import subprocess_utils
from navigator_updater.utils.conda import get_conda_info
from navigator_updater.utils.conda.core import get_conda_cmd_path
from navigator_updater.utils.encoding import ensure_binary
from navigator_updater.utils.logs.loggers import conda_logger as logger
from navigator_updater.utils.toolpath import get_pyexec, get_pyscript
from . import repodata


class ConfigIndexContent(typing.TypedDict):
    """Structure of the config index data structures."""

    current: typing.Optional[str]
    snapshots: typing.Dict[str, str]


__version__ = '1.3.0'

if WIN:
    import ctypes

# --- Constants
# -----------------------------------------------------------------------------
CONDA_API = None
RC_STORAGE_DIR = os.path.join(CONF_PATH, 'condarc_storage')
DEFAULT_RC_KEY = 'main'


# --- Errors
# -----------------------------------------------------------------------------


class CondaError(Exception):
    """General Conda error."""


class CondaProcessWorker(CondaError):
    """General Conda error."""


class CondaEnvExistsError(CondaError):
    """Conda environment already exists."""


# --- Helpers
# -----------------------------------------------------------------------------

def to_text_string(obj, encoding=None):
    """Convert `obj` to (unicode) text string."""
    if encoding is None:
        return str(obj)
    if isinstance(obj, str):
        # In case this function is not used properly, this could happen
        return obj
    return str(obj, encoding)


def handle_qbytearray(obj, encoding):
    """Qt/Python3 compatibility helper."""
    if isinstance(obj, QByteArray):
        obj = obj.data()
    return to_text_string(obj, encoding=encoding)


class ProcessWorker(QObject):  # pylint: disable=too-many-instance-attributes
    """Conda worker based on a QProcess for non blocking UI."""

    sig_chain_finished = Signal(object, object, object)
    sig_finished = Signal(object, object, object)
    sig_partial = Signal(object, object, object)

    def __init__(self, cmd_list, parse=False, pip=False, callback=None, extra_kwargs=None, environ=None):
        """Conda worker based on a QProcess for non blocking UI.

        Parameters
        ----------
        cmd_list : list of str
            Command line arguments to execute.
        parse : bool (optional)
            Parse json from output.
        pip : bool (optional)
            Define as a pip command.
        callback : func (optional)
            If the process has a callback to process output from comd_list.
        extra_kwargs : dict
            Arguments for the callback.
        """
        super().__init__()
        self._result = None
        self._cmd_list = cmd_list
        self._parse = parse
        self._pip = pip
        self._conda = not pip
        self._callback = callback
        self._fired = False
        self._communicate_first = False
        self._partial_stdout = None
        self._extra_kwargs = extra_kwargs if extra_kwargs else {}

        self._timer = QTimer()
        self._process = QProcess()
        self._set_environment(environ)

        self._timer.setInterval(150)

        self._timer.timeout.connect(self._communicate)
        # self._process.finished.connect(self._communicate)
        self._process.readyReadStandardOutput.connect(self._partial)

    @staticmethod
    def get_encoding():
        """Return the encoding/codepage to use."""
        enco = 'utf-8'

        #  Currently only cp1252 is allowed
        if WIN:
            codepage = str(ctypes.cdll.kernel32.GetACP())
            enco = 'cp' + codepage

        return enco

    def _set_environment(self, environ):
        """Set the environment on the QProcess."""
        if environ:
            q_environ = self._process.processEnvironment()
            for k, v in environ.items():
                q_environ.insert(k, v)

            # '0' for CONDA_SHLVL indicates that the conda shell function is
            # available within the shell, but there is currently no environment activated
            if 'CONDA_SHLVL' in environ and WIN:
                q_environ.insert('CONDA_SHLVL', '0')
            self._process.setProcessEnvironment(q_environ)

    def _partial(self):
        """Callback for partial output."""
        # if self._process != QProcess.NotRunning:
        raw_stdout = self._process.readAllStandardOutput()
        stdout = handle_qbytearray(raw_stdout, self.get_encoding())

        try:
            json_stdout = [json.loads(s) for s in stdout.split('\x00') if s]
            json_stdout = json_stdout[-1]
            json_stdout = self.adjust_pip_dict(json_stdout)
        except Exception:  # pylint: disable=broad-except
            json_stdout = stdout

        if self._partial_stdout is None:
            self._partial_stdout = stdout
        else:
            self._partial_stdout += stdout

        self.sig_partial.emit(self, json_stdout, None)

    def _communicate(self):
        """Callback for communicate."""
        if (not self._communicate_first and self._process.state() == QProcess.NotRunning):
            self.communicate()
        elif self._fired:
            self._timer.stop()

    def adjust_pip_dict(self, content):
        """Convert pip output to internal used dict."""
        if self._pip is not True:
            return content
        result = {}
        for d in content:
            name = d.get('name', '')
            ver = d.get('version', '')
            full_name = f'{name.lower()}-{ver}-pip'
            result[full_name] = {'version': ver}
        return result

    def communicate(self):  # pylint: disable=too-many-branches,too-many-locals
        """Retrieve information."""
        self._communicate_first = True
        self._process.waitForFinished()

        enco = self.get_encoding()
        if self._partial_stdout is None:
            raw_stdout = self._process.readAllStandardOutput()
            stdout = handle_qbytearray(raw_stdout, enco)
        else:
            stdout = self._partial_stdout

        raw_stderr = self._process.readAllStandardError()
        stderr = handle_qbytearray(raw_stderr, enco)
        stdout = ansi_utlils.escape_ansi(stdout)
        stderr = ansi_utlils.escape_ansi(stderr)
        result = [stdout.encode(enco), stderr.encode(enco)]

        # NOTE: Why does anaconda client print to stderr???

        result[-1] = ''

        if self._parse and stdout:
            json_stdout = []
            json_lines_output = stdout.split('\x00')
            for i, l in enumerate(json_lines_output):
                left: int = 0
                character: str
                for left, character in enumerate(l):
                    if character in '[{':
                        break
                right: int = 0
                for right in reversed(range(len(l))):
                    if l[right] in ']}':
                        break
                if right >= left:
                    try:
                        jd = self.adjust_pip_dict(json.loads(l[left:right + 1]))
                        json_stdout.append(jd)
                    except Exception as error:  # pylint: disable=broad-except
                        # An exception here could be product of:
                        # - conda env installing pip stuff that is thrown to
                        #   stdout in non json form
                        # - a post link script might be printing stuff to
                        #   stdout in non json format
                        logger.warning(
                            'Problem parsing conda json output. Line %s. Data - %s. Error - %s', i, l, str(error),
                        )

            if json_stdout:
                json_stdout = json_stdout[-1]
            result = json_stdout, result[-1]

            out = result[0]
            if 'exception_name' in out or 'exception_type' in out:
                if not isinstance(out, dict):
                    result = {'error': str(out)}, None
                else:
                    result = out, f'{" ".join(self._cmd_list)}: {out.get("message", "")}'

        if self._callback:
            result = self._callback(result[0], stderr, **self._extra_kwargs), result[-1]

        self._result = result

        # NOTE: Remove chained signals and use a _sig_finished
        self.sig_finished.emit(self, result[0], result[-1])

        self._fired = True
        logger.debug('[CONDA] %s', ' '.join(self._cmd_list), extra={
            'output': result[0],
            'error': result[-1],
            'environment': dict(os.environ),
            'callback': str(self._callback)
        })
        return result

    def close(self):
        """Close the running process."""
        self._process.close()

    def is_finished(self):
        """Return True if worker has finished processing."""
        return self._process.state() == QProcess.NotRunning and self._fired

    def start(self):
        """Start process."""
        if not self._fired:
            self._partial_ouput = None  # pylint: disable=attribute-defined-outside-init
            self._process.start(self._cmd_list[0], self._cmd_list[1:])
            self._timer.start()
        else:
            raise CondaProcessWorker('A Conda ProcessWorker can only run once per method call.')


class ConfigIndex:
    """Key-value storage for tracking condarc copies."""

    index_path: typing.Final[str] = os.path.join(RC_STORAGE_DIR, 'condarc_index.json')

    def __init__(self, old_config_copy_path=None):
        """Create folder for copies of condarc and index file if missing. Else load index from `json` file on disc"""
        self.__index: typing.Final[ConfigIndexContent] = {
            'current': None,
            'snapshots': {},
        }
        if os.path.exists(self.index_path):
            self._load()

        if old_config_copy_path:
            self.__move_old_copy_to_index(old_config_copy_path)

    @property
    def current(self) -> typing.Optional[str]:  # noqa: D401
        """Key of the current .condarc file."""
        return self.__index['current']

    @current.setter
    def current(self, value: typing.Optional[str]) -> None:
        """Update `current` value."""
        self.__index['current'] = value
        self._save()

    def __getitem__(self, key: str) -> typing.Optional[str]:
        """Get value from the index"""
        return self.__index['snapshots'].get(key, None)

    def __setitem__(self, key: str, value: str) -> None:
        """Add key-value record to the index and save index on disc"""
        self.__index['snapshots'][key] = value
        self._save()

    def __add_path(self, rc_key: str) -> str:
        """Generate hash from key, construct full path to a copy of condarc file and add key/path pair to the index"""
        filename: str = f'.condarc_{hashlib.md5(ensure_binary(rc_key)).hexdigest()}'  # nosec
        path: str = os.path.join(RC_STORAGE_DIR, filename)

        if self.__index['snapshots'].get(rc_key) != path:
            self[rc_key] = path

        return path

    def __move_old_copy_to_index(self, old_copy_path: str) -> None:
        index_copy_path = self.__add_path(DEFAULT_RC_KEY)

        if os.path.isfile(old_copy_path):
            os.rename(old_copy_path, index_copy_path)

    def _save(self) -> None:
        """Save index from `json` file on disc"""
        os.makedirs(os.path.dirname(os.path.abspath(self.index_path)), exist_ok=True)

        stream: typing.TextIO
        with open(self.index_path, 'wt', encoding='utf-8') as stream:
            json.dump(self.__index, stream)

    def _load(self) -> None:
        """Load index from disc as `json` file"""
        content: typing.Any
        try:
            stream: typing.TextIO
            with open(self.index_path, 'rt', encoding='utf-8') as stream:
                content = json.load(stream)
        except (OSError, ValueError, TypeError):
            # do not break the application if the file is broken
            return
        self.__index.update(content)

    @staticmethod
    def generate_config_index_key(*keys: str) -> str:
        return '_'.join(
            re.sub(r'\W+', '_', key.lower())
            for key in keys
        )

    def save_rc_copy(self, data: typing.Mapping[str, typing.Any], rc_key: str = DEFAULT_RC_KEY) -> None:
        stream: typing.TextIO
        path: str = os.path.abspath(self.__add_path(rc_key))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wt', encoding='utf-8') as stream:
            yaml.dump(data, stream)

        logger.debug('Config file with rc_key `%s` was saved.', rc_key)

    def load_rc_copy(self, rc_key: str = DEFAULT_RC_KEY) -> typing.Mapping[str, typing.Any]:
        path: typing.Optional[str] = self[rc_key]
        if (not path) or (not os.path.isfile(path)):
            return {}

        stream: typing.TextIO
        with open(path, 'rt', encoding='utf-8') as stream:
            rc_data = yaml.full_load(stream)
            logger.debug('Config file with rc_key `%s` was loaded.', rc_key)
            return rc_data


# --- API
# -----------------------------------------------------------------------------
class _CondaAPI(QObject):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Conda API to connect to conda in a non blocking way via QProcess."""

    def __init__(self, parent=None):
        """Conda API to connect to conda in a non blocking way via QProcess."""
        super().__init__()

        # Variables
        self._parent = parent
        self._queue = deque()
        self._timer = QTimer()
        self._current_worker = None
        self._workers = []

        # Conda config values
        self.CONDA_PREFIX = None
        self.ROOT_PREFIX: typing.Optional[str] = None
        self.CONDA_EXE: typing.Optional[str] = None
        self._envs_dirs = None
        self._pkgs_dirs = None
        self._user_agent = None
        self._proxy_servers = None
        self._conda_version = None

        self.set_conda_prefix(info=get_conda_info())

        self.user_rc_path = os.path.abspath(os.path.expanduser('~/.condarc'))
        if self.ROOT_PREFIX is None:
            self.sys_rc_path = None
        else:
            self.sys_rc_path = os.path.join(self.ROOT_PREFIX, '.condarc')

        self.__rc_index: typing.Final[ConfigIndex] = ConfigIndex(
            old_config_copy_path=self.get_old_config_copy_path(),
        )

        # Setup
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._clean)

        # Cache
        self._app_info_cache = {}

    @property
    def rc_index(self) -> ConfigIndex:
        """Index of the historical condarc """
        return self.__rc_index

    def _clean(self):
        """Remove references of inactive workers periodically."""
        if self._workers:
            for w in self._workers:
                if w.is_finished():
                    self._workers.remove(w)
        else:
            self._current_worker = None
            self._timer.stop()

    def _start(self):
        if len(self._queue) == 1:
            self._current_worker = self._queue.popleft()
            self._workers.append(self._current_worker)
            self._current_worker.start()
            self._timer.start()

    def terminate_all_processes(self):
        """Kill all working processes."""
        for worker in self._workers:
            # Try to disconnect signals first
            with contextlib.suppress(BaseException):
                worker.sig_finished.disconnect()
            with contextlib.suppress(BaseException):
                worker.sig_partial.disconnect()
            # Now close the worker
            worker.close()

    # --- Conda api
    # -------------------------------------------------------------------------
    def _call_conda(self, extra_args, parse=False, callback=None, environ=None):
        """
        Call conda with the list of extra arguments, and return the worker.

        The result can be force by calling worker.communicate(), which returns
        the tuple (stdout, stderr).
        """
        conda_exe: typing.Optional[str] = self.CONDA_EXE
        if conda_exe is None:
            if self.ROOT_PREFIX is None:
                conda_exe = get_conda_cmd_path()
            else:
                conda_exe = get_conda_cmd_path(self.ROOT_PREFIX)

        process_worker = ProcessWorker(
            [conda_exe, *extra_args],
            parse=parse,
            callback=callback,
            environ=environ,
        )
        process_worker.sig_finished.connect(self._start)
        self._queue.append(process_worker)
        self._start()

        return process_worker

    def _call_and_parse(self, extra_args, callback=None, environ=None):
        return self._call_conda(
            extra_args,
            parse=True,
            callback=callback,
            environ=environ,
        )

    @staticmethod
    def _setup_install_commands_from_kwargs(kwargs, keys=tuple()):
        """Setup install commands for conda."""
        cmd_list = []
        if kwargs.get('override_channels', False) and 'channel' not in kwargs:
            raise TypeError('conda search: override_channels requires channel')

        if 'env' in kwargs:
            cmd_list.extend(['--name', kwargs.pop('env')])
        if 'prefix' in kwargs:
            cmd_list.extend(['--prefix', kwargs.pop('prefix')])
        if 'channel' in kwargs:
            channel = kwargs.pop('channel')
            if isinstance(channel, str):
                cmd_list.extend(['--channel', channel])
            else:
                cmd_list.append('--channel')
                cmd_list.extend(channel)

        for key in keys:
            if key in kwargs and kwargs[key]:
                cmd_list.append('--' + key.replace('_', '-'))

        return cmd_list

    def _set_environment_variables(self, prefix=None, no_default_python=False):
        """Set the right CONDA_PREFIX environment variable."""
        environ_copy = os.environ.copy()
        conda_prefix = self.ROOT_PREFIX
        if prefix:
            conda_prefix = prefix

        if conda_prefix:
            if conda_prefix == self.ROOT_PREFIX:
                name = 'root'
            else:
                name = os.path.basename(conda_prefix)
            environ_copy['CONDA_PREFIX'] = conda_prefix
            environ_copy['CONDA_DEFAULT_ENV'] = name

        if no_default_python:
            environ_copy['CONDA_DEFAULT_PYTHON'] = None

        # NOTE: this is meant to short-circuit failure, such that unsatisfiable errors for rstudio happen faster.
        #  As of conda 4.8.2, this is erroring out even when the environment is not fully determined to be unsat.
        #  This might be useful in the future, but it will require changes on the conda side.

        return environ_copy

    def set_conda_prefix(self, info=None):
        """
        Set the prefix of the conda environment.

        This function should only be called once (right after importing
        conda_api).
        """
        if info is None:
            # Find some conda instance, and then use info to get 'root_prefix'
            worker = self.info()
            info = worker.communicate()[0]

        self.ROOT_PREFIX = info.get('root_prefix')  # str
        self.CONDA_PREFIX = info.get('conda_prefix')  # str
        self._envs_dirs = info.get('envs_dirs')  # Sequence[str]
        self._pkgs_dirs = info.get('pkgs_dirs')  # Sequence[str]
        self._user_agent = info.get('user_agent')  # str

        self.CONDA_EXE = get_conda_cmd_path(self.ROOT_PREFIX)

        version = []
        for part in info.get('conda_version').split('.'):
            try:
                new_part = int(part)
            except ValueError:
                new_part = part

            version.append(new_part)

        self._conda_version = tuple(version)

    def get_conda_version(self):
        """Return the version of conda being used (invoked) as a string."""
        return self._call_conda(
            ['--version'],
            callback=self._get_conda_version,
        )

    @staticmethod
    def _get_conda_version(stdout, stderr):
        """Callback for get_conda_version."""
        # argparse outputs version to stderr in Python < 3.4.
        # http://bugs.python.org/issue18920
        pat = re.compile(r'conda:?\s+(\d+\.\d\S+|unknown)')
        try:
            m = pat.match(stderr.decode().strip())
        except Exception:  # pylint: disable=broad-except
            m = pat.match(stderr.strip())

        if m is None:
            try:
                m = pat.match(stdout.decode().strip())
            except Exception:  # pylint: disable=broad-except
                m = pat.match(stdout.strip())

        if m is None:
            raise Exception(f'output did not match: {stderr}')   # pylint: disable=broad-exception-raised

        return m.group(1)

    @property
    def pkgs_dirs(self) -> typing.Sequence[str]:
        """Conda package cache directories."""
        if self._pkgs_dirs:
            return self._pkgs_dirs

        result: typing.List[str] = []

        if self.ROOT_PREFIX is not None:
            result.append(os.path.join(self.ROOT_PREFIX, 'pkgs'))

        result.append(os.path.join(get_home_dir(), '.conda', 'pkgs'))

        return result

    @property
    def envs_dirs(self):
        """
        Conda environment directories.

        The first writable item should be used.
        """
        if self._envs_dirs:
            envs_dirs = self._envs_dirs
        else:
            # Legacy behavior
            envs_path = os.sep.join([self.ROOT_PREFIX, 'envs'])
            user_envs_path = os.sep.join([get_home_dir(), '.conda', 'envs'])
            envs_dirs = [envs_path, user_envs_path]

        return envs_dirs

    @property
    def user_agent(self):
        return self._user_agent

    def get_envs(self, log=True):
        """Return environment list of absolute path to their prefixes."""
        all_envs = []
        for env in self.envs_dirs:
            if os.path.isdir(env):
                envs_names = os.listdir(env)
                all_envs += [os.sep.join([env, i]) for i in envs_names]

        valid_envs = [env for env in all_envs if os.path.isdir(env) and self.environment_exists(prefix=env)]

        return valid_envs

    def get_prefix_envname(self, name, log=None):
        """Return full prefix path of environment defined by `name`."""
        prefix = None
        if name == 'root':
            prefix = self.ROOT_PREFIX

        envs = self.get_envs()
        for p in envs:
            if os.path.basename(p) == name:
                prefix = p

        return prefix

    def get_name_envprefix(self, prefix):
        """
        Return name of environment defined by full `prefix` path.

        Returns the name if it is located in the default conda environments
        directory, otherwise it returns the prefix.
        """
        name = os.path.basename(prefix)
        if not (name and self.environment_exists(name=name)):
            name = prefix
        return name

    @staticmethod
    def get_local_channel_data(channel: str) -> typing.Dict[str, typing.Any]:
        channel_data: typing.Dict[str, typing.Any] = {}

        if not channel.startswith('file:///'):
            return channel_data

        channel = misc.convert_file_url_to_path(channel)
        channel_data_path: str = os.path.join(channel, 'channeldata.json')

        try:
            with open(channel_data_path, encoding='utf-8') as f:
                channel_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.exception(e)

        return channel_data

    @staticmethod
    def linked(prefix: str) -> typing.Set[str]:
        """Return set of canonical names of linked packages in `prefix`."""
        if not os.path.isdir(prefix):
            return set()

        packages: typing.Set[str] = set()
        meta_dir: str = os.path.join(prefix, 'conda-meta')

        if os.path.isdir(meta_dir):
            meta_files: typing.Set[str] = set(fname for fname in os.listdir(meta_dir) if fname.endswith('.json'))
            packages = set(fname[:-5] for fname in meta_files)

        return packages

    @staticmethod
    def get_icon_path_from_recipe(package_path: str) -> typing.Optional[str]:
        recipe_dir: typing.Final[str] = os.path.join(package_path, 'info', 'recipe')
        recipe_meta: typing.Final[str] = os.path.join(recipe_dir, 'meta.yaml')

        if not os.path.isfile(recipe_meta):
            return None

        try:
            with open(recipe_meta, encoding='utf-8') as f:
                recipe: typing.Mapping[str, typing.Any] = yaml.safe_load(f)
        except (yaml.YAMLError, OSError) as e:
            logger.exception(e)

        icon_name: str = recipe.get('app', {}).get('icon', '')
        icon_path: str = os.path.join(recipe_dir, icon_name)

        if icon_name and os.path.exists(icon_path):
            return icon_path

        return None

    def local_apps_info(self):
        apps_info: typing.Dict[api_types.ApplicationName, 'api_types.RawApplication'] = {}

        channels: typing.List[str] = self.load_rc().get('channels', [])
        channel_data: typing.Dict[str, typing.Dict[str, typing.Any]] = {}

        for channel in channels:
            data: typing.Dict[str, typing.Any] = self.get_local_channel_data(channel)
            if data:
                channel_data[channel] = data

        for channel,  data in channel_data.items():
            package_name: api_types.ApplicationName
            package_data: typing.Dict[str, typing.Any]
            for package_name, package_data in data.get('packages', {}).items():
                version: str = package_data.get('version', '')
                image_path: str = os.path.join(misc.convert_file_url_to_path(channel), package_data.get('icon_url', ''))
                app_data: 'api_types.RawApplication' = {
                    'name': package_name,
                    'description': package_data.get('summary', ''),
                    'versions': [version],
                    'image_path': image_path if os.path.exists(image_path) else None
                }
                apps_info[package_name] = app_data

        return apps_info

    def linked_apps_info(
            self, prefix: str,
            envs_cache: bool = False) -> typing.Dict[api_types.ApplicationName, 'api_types.RawApplication']:
        """Return local installed apps info on prefix."""
        linked_pkgs: typing.Set[str] = self.linked(prefix)
        pkgs_dirs: typing.List[str] = [*(self.pkgs_dirs or [])]
        if envs_cache:
            pkgs_dirs.extend((os.path.join(env, 'pkgs') for env in self.get_envs()))

        directory: str
        linked_pkg: str
        for directory, linked_pkg in itertools.product(pkgs_dirs, linked_pkgs):
            fpath: str = os.path.join(directory, linked_pkg, 'info', 'index.json')
            if fpath in self._app_info_cache:
                continue

            if not os.path.isfile(fpath):
                continue

            with open(fpath) as f:
                data: typing.Mapping[str, typing.Any] = json.load(f)

            app_info: 'api_types.RawApplication' = {}
            if 'app_entry' in data or 'app_type' in data:
                name: str
                version: str
                name, version, _ = split_canonical_name(linked_pkg)

                app_info = {
                    'name': data.get('name', name),
                    'description': data.get('summary', ''),
                    'versions': [version],
                    'app_entry': {
                        version: data.get('app_entry', '')
                    },
                    'image_path': self.get_icon_path_from_recipe(os.path.join(directory, linked_pkg)) or '',
                }
            self._app_info_cache[fpath] = app_info

        return {_['name']: _ for _ in self._app_info_cache.values() if _}

    def info(
        self,
        prefix=None,
        unsafe_channels=False,
        all_=False,
    ):
        """Return a dictionary with configuration information."""
        environ = self._set_environment_variables(prefix)
        cmd_list = ['info', '--json']

        if unsafe_channels:
            cmd_list.extend(['--unsafe-channels'])

        if all_:
            cmd_list.extend(['--all'])

        return self._call_and_parse(cmd_list, environ=environ)

    def search(self, regex=None, spec=None, prefix=None, offline=False, **kwargs):
        """Search for packages."""
        cmd_list = ['search', '--json']
        environ = self._set_environment_variables(prefix)

        if regex and spec:
            raise TypeError('conda search: only one of regex or spec allowed')

        if regex:
            cmd_list.append(regex)

        if spec:
            cmd_list.extend(['--spec', spec])

        if 'platform' in kwargs:
            cmd_list.extend(['--platform', kwargs.pop('platform')])

        cmd_list.extend(
            self._setup_install_commands_from_kwargs(
                kwargs, (
                    'canonical',
                    'unknown',
                    'use_index_cache',
                    'outdated',
                    'override_channels',
                )
            )
        )

        if offline:
            cmd_list.extend(['--offline'])

        return self._call_and_parse(cmd_list, environ=environ)

    def list(self, regex=None, prefix=None, **kwargs):
        """List packages packages."""
        cmd_list = ['list', '--json']
        environ = self._set_environment_variables(prefix)

        if regex:
            cmd_list.append(regex)

        cmd_list.extend(
            self._setup_install_commands_from_kwargs(
                kwargs, (
                    'canonical',
                    'show_channel_urls',
                    'full_name',
                    'explicit',
                    'revision',
                    'no_pip'
                )
            )
        )

        return self._call_and_parse(cmd_list, environ=environ)

    # --- Conda Environment Actions
    # -------------------------------------------------------------------------
    def export_environment(  # pylint: disable=too-many-locals
            self,
            file: str,
            name: typing.Optional[str] = None,
            prefix: typing.Optional[str] = None,
            ignore_channels: bool = False,
            override_channels: bool = False,
            extra_channels: typing.Iterable[str] = (),
            no_builds: bool = False,
            from_history: bool = False,
    ) -> ProcessWorker:
        """
        Export conda environment to a yaml file.

        Environment must be provided by either `name` or `prefix`.

        :param file: Path to a file, where to store exported environment details.
        :param name: Name of the environment.
        :param prefix: Full path to environment location (i.e. prefix).
        :param ignore_channels: Do not include channel names with package names.
        :param override_channels: Do not include .condarc channels
        :param extra_channels: Additional channel(s) to include in the export.
        :param no_builds: Remove build specification from dependencies.
        :param from_history: Build environment spec from explicit specs in history.
        :return: Worker for the environment export request.
        """
        # Argument validations
        if not file:
            raise ValueError('`value` can not be empty')

        if sum(map(bool, (name, prefix))) != 1:
            raise TypeError('either `name` or `prefix` must be set')

        # Base command
        cmd_list: typing.List[str] = ['env', 'export']

        # Flags
        key: str
        flag: bool
        flags: typing.Mapping[str, bool] = {
            '--override-channels': override_channels,
            '--ignore-channels': ignore_channels,
            '--from-history': from_history,
            '--no-builds': no_builds,
        }
        for key, flag in flags.items():
            if flag:
                cmd_list.append(key)

        # Values
        value: typing.Optional[str]
        values: typing.Mapping[str, typing.Optional[str]] = {
            '--file': file,
            '--name': name,
            '--prefix': prefix,
        }
        for key, value in values.items():
            if value:
                cmd_list.extend((key, value))

        # Arrays
        array: typing.Iterable[str]
        arrays: typing.Mapping[str, typing.Iterable[str]] = {
            '--channel': extra_channels,
        }
        for key, array in arrays.items():
            for value in array:
                if value:
                    cmd_list.extend((key, value))

        # Execution
        return self._call_conda(cmd_list)  # without parsing, as `--json` breaks this command

    def _create_from_yaml(self, yamlfile, prefix=None, name=None, offline=False, dry_run=False):
        """
        Create new environment using conda-env via a yaml specification file.

        Unlike other methods, this calls conda-env, and requires a named
        environment and uses channels as defined in rcfiles.

        Parameters
        ----------
        name : string
            Environment name
        yamlfile : string
            Path to yaml file with package spec (as created by conda env export
        """
        if (name is None) and (prefix is None):
            raise TypeError('must specify a `name` or `prefix`')

        cmd_list = ['env', 'create', '-f', yamlfile, '--json']
        if name:
            cmd_list.extend(['--name', name])
        elif prefix:
            cmd_list.extend(['--prefix', prefix])

        if offline:
            cmd_list.append('--offline')

        if dry_run:
            cmd_list.append('--dry-run')

        return self._call_and_parse(cmd_list)

    def create(  # pylint: disable=too-many-branches
        self,
        name: typing.Optional[str] = None,
        prefix: typing.Optional[str] = None,
        pkgs: typing.Optional[typing.Sequence[str]] = None,
        file: typing.Optional[str] = None,
        no_default_python: bool = False,
        offline: bool = False,
        dry_run: bool = False,
    ) -> ProcessWorker:
        """
        Create an environment with a specified set of packages.

        Default python option is on deprecation route for 4.4.
        """
        environ = self._set_environment_variables(prefix=prefix, no_default_python=no_default_python)

        if (name is None) and (prefix is None):
            raise TypeError('must specify a `name` or `prefix`')

        if file and file.endswith(('.yaml', '.yml')):
            result = self._create_from_yaml(file, prefix=prefix, name=name, offline=offline, dry_run=dry_run)
        else:
            if (not pkgs) and (not file):
                raise TypeError('must specify a list of one or more packages to install into new environment')

            # mkdir removed in conda 4.6
            if self._conda_version > (4, 5):
                cmd_list = ['create', '--yes', '--json']
            else:
                cmd_list = ['create', '--yes', '--json', '--mkdir']

            # Explicit conda file spec provided
            if file and file.endswith(('.txt', )):
                cmd_list.extend(['--file', file])

            if name:
                ref = name
                search = [os.path.join(d, name) for d in self.envs_dirs]
                cmd_list.extend(['--name', name])
            elif prefix:
                ref = prefix
                search = [prefix]
                cmd_list.extend(['--prefix', prefix])
            else:
                raise TypeError('must specify either an environment name or a path for new environment')

            if any(os.path.exists(prefix) for prefix in search):
                raise CondaEnvExistsError(f'Conda environment {ref} already exists')

            # If file spec provided, this should be None
            if pkgs:
                cmd_list.extend(pkgs)

            if offline:
                cmd_list.append('--offline')

            if dry_run:
                cmd_list.append('--dry-run')

            result = self._call_and_parse(cmd_list, environ=environ)

        return result

    def install(
        self,
        name=None,
        prefix=None,
        pkgs=None,
        dep=True,
        dry_run=False,
        no_default_python=False,
        offline=False,
    ):
        """Install a set of packages into an environment by name or path."""
        environ = self._set_environment_variables(prefix=prefix, no_default_python=no_default_python)

        # NOTE: Fix temporal hack
        if not pkgs or not isinstance(pkgs, (list, tuple, str)):
            raise TypeError('must specify a list of one or more packages to install into existing environment')

        cmd_list = ['install', '--yes', '--json', '--force-pscheck']
        if name:
            cmd_list.extend(['--name', name])
        elif prefix:
            cmd_list.extend(['--prefix', prefix])
        else:
            # Just install into the current environment, whatever that is
            pass

        # NOTE: Fix temporal hack
        if isinstance(pkgs, (list, tuple)):
            cmd_list.extend(pkgs)
        elif isinstance(pkgs, str):
            cmd_list.extend(['--file', pkgs])

        if not dep:
            cmd_list.extend(['--no-deps'])

        if dry_run:
            cmd_list.extend(['--dry-run'])

        if offline:
            cmd_list.extend(['--offline'])

        return self._call_and_parse(cmd_list, environ=environ)

    def update(
        self,
        name=None,
        prefix=None,
        pkgs=None,
        dep=True,
        all_=False,
        dry_run=False,
        no_default_python=False,
        offline=False,
    ):
        """Install a set of packages into an environment by name or path."""
        environ = self._set_environment_variables(prefix=prefix, no_default_python=no_default_python)

        cmd_list = ['update', '--yes', '--json', '--force-pscheck']
        if not pkgs and not all_:
            raise TypeError('Must specify at least one package to update, or all_=True.')

        if name:
            cmd_list.extend(['--name', name])
        elif prefix:
            cmd_list.extend(['--prefix', prefix])
        else:
            # Just install into the current environment, whatever that is
            pass

        if isinstance(pkgs, (list, tuple)):
            cmd_list.extend(pkgs)

        if not dep:
            cmd_list.extend(['--no-deps'])

        if dry_run:
            cmd_list.extend(['--dry-run'])

        if offline:
            cmd_list.extend(['--offline'])

        return self._call_and_parse(cmd_list, environ=environ)

    def remove(
        self,
        name=None,
        prefix=None,
        pkgs=None,
        all_=False,
        dry_run=False,
        offline=False,
    ):
        """
        Remove a package (from an environment) by name.

        Returns {
            success: bool, (this is always true),
            (other information)
        }
        """
        cmd_list = ['remove', '--json', '--yes']

        if not pkgs and not all_:
            raise TypeError('Must specify at least one package to remove, or all=True.')

        if name:
            cmd_list.extend(['--name', name])
        elif prefix:
            cmd_list.extend(['--prefix', prefix])
        else:
            raise TypeError('must specify either an environment name or a path for package removal')

        if all_:
            cmd_list.extend(['--all'])
        else:
            cmd_list.extend(pkgs)

        if dry_run:
            cmd_list.extend(['--dry-run'])

        if offline:
            cmd_list.extend(['--offline'])

        return self._call_and_parse(cmd_list)

    def remove_environment(self, name=None, prefix=None, offline=False):
        """Remove an environment entirely specified by `name` or `prefix`."""
        return self.remove(name=name, prefix=prefix, all_=True, offline=offline)

    def clone_environment(self, clone_from_prefix, name=None, prefix=None, offline=False, **kwargs):
        """Clone the environment `clone` into `name` or `prefix`."""
        cmd_list = ['create', '--json']

        if (name and prefix) or not (name or prefix):
            raise TypeError('conda clone_environment: exactly one of `name` or `path` required')

        if name:
            cmd_list.extend(['--name', name])

        if prefix:
            cmd_list.extend(['--prefix', prefix])

        cmd_list.extend(['--clone', clone_from_prefix])

        cmd_list.extend(
            self._setup_install_commands_from_kwargs(
                kwargs, (
                    'dry_run',
                    'unknown',
                    'use_index_cache',
                    'use_local',
                    'no_pin',
                    'force',
                    'all',
                    'channel',
                    'override_channels',
                    'no_default_packages',
                )
            )
        )

        if offline:
            cmd_list.extend(['--offline'])

        return self._call_and_parse(cmd_list)

    # --- Conda Configuration
    # -------------------------------------------------------------------------
    @staticmethod
    def _setup_config_from_kwargs(kwargs):
        """Setup config commands for conda."""
        cmd_list = ['--json', '--force']

        if 'file' in kwargs:
            cmd_list.extend(['--file', kwargs['file']])

        if 'system' in kwargs:
            cmd_list.append('--system')

        return cmd_list

    def _setup_config_args(self, file=None, prefix=None, system=False):
        """Setup config commands for conda."""
        cmd_list = ['--json', '--force']

        if file:
            config_file = file
        elif prefix and self.environment_exists(prefix):
            config_file = os.path.join(self.ROOT_PREFIX, '.condarc')
        elif system:
            config_file = self.sys_rc_path
        else:
            config_file = self.user_rc_path

        cmd_list.extend(['--file', config_file])

        return cmd_list

    def config_get(self, *keys, **kwargs):
        """
        Get the values of configuration keys.

        Returns a dictionary of values. Note, the key may not be in the
        dictionary if the key wasn't set in the configuration file.
        """
        cmd_list = ['config', '--get']
        cmd_list.extend(keys)
        cmd_list.extend(self._setup_config_from_kwargs(kwargs))

        return self._call_and_parse(cmd_list, callback=lambda o, e: o['get'])

    def config_set(self, key, value, file=None, prefix=None, system=False):
        """
        Set a key to a (bool) value.

        Returns a list of warnings Conda may have emitted.
        """
        cmd_list = ['config', '--set', key, str(value)]
        args = self._setup_config_args(system=system, file=file, prefix=prefix)
        cmd_list.extend(args)

        return self._call_and_parse(cmd_list, callback=lambda o, e: o.get('warnings', []))

    def config_add(self, key, value, file=None, prefix=None, system=False):
        """
        Add a value to a key.

        Returns a list of warnings Conda may have emitted.
        """
        cmd_list = ['config', '--add', key, value]
        args = self._setup_config_args(system=system, file=file, prefix=prefix)
        cmd_list.extend(args)

        return self._call_and_parse(cmd_list, callback=lambda o, e: o.get('warnings', []))

    def config_remove(self, key, value, file=None, prefix=None, system=False):
        """
        Remove a value from a key.

        Returns a list of warnings Conda may have emitted.
        """
        cmd_list = ['config', '--remove', key, value]
        args = self._setup_config_args(system=system, file=file, prefix=prefix)
        cmd_list.extend(args)

        return self._call_and_parse(cmd_list, callback=lambda o, e: o.get('warnings', []))

    def config_delete(self, key, file=None, prefix=None, system=False):
        """
        Remove a key entirely.

        Returns a list of warnings Conda may have emitted.
        """
        cmd_list = ['config', '--remove-key', key]
        args = self._setup_config_args(system=system, file=file, prefix=prefix)
        cmd_list.extend(args)

        return self._call_and_parse(cmd_list, callback=lambda o, e: o.get('warnings', []))

    @staticmethod
    def _config_show_sources(sources, error, prefix=None, all_=False):
        """Callback for show sources method."""
        file_base_sources = {}
        if 'cmd_line' in sources:
            sources.pop('cmd_line')

            for k, v in sources.items():
                if os.path.isfile(k):
                    file_base_sources[k] = v

        return file_base_sources

    def config_show_sources(self, prefix=None, all_=False):
        """
        Show configuration sources.

        Parameters
        ----------
        prefix : str
            This is equivalent of using `--env` flag for the activated
            environent `prefix`.
        all : bool
            This includes all the configuration options in envs, which depend
            on the concept of an activated environment. If both prefix and
            all are provided, all overides the specific path.
        """
        return self.config_show(sources=True, prefix=prefix, all_=all_)

    def config_show(self, prefix=None, all_=False, sources=False):
        """Show configuration options."""
        cmd_list = ['config', '--json']

        environ = self._set_environment_variables(prefix=prefix)
        if sources:
            cmd_list.append('--show-sources')
            worker = self._call_and_parse(
                cmd_list,
                callback=lambda o, e: self._config_show_sources(o, e, prefix=prefix, all_=all_),
                environ=environ,
            )
        else:
            cmd_list.append('--show')
            worker = self._call_and_parse(cmd_list, environ=environ)
        return worker

    def get_old_config_copy_path(self):
        config_path = self.generate_config_path()
        config_path_copied = f'{config_path}_copy'

        if os.path.exists(config_path_copied):
            return config_path_copied

        return None

    def generate_config_path(self, path=None, prefix=None, system=False):
        """
        Generates valid config path depending on the passed attributes for configuration.

        Parameters
        ----------
        path : str
            Path to conda configuration file.
        prefix : str
            Prefix path, to retrieve the specific prefix configuration file.
        system : bool
            Retrieve the system configuration file.
        """
        if path:
            return path
        if prefix and self.environment_exists(prefix=prefix):
            return os.path.join(prefix, '.condarc')
        if system:
            return self.sys_rc_path
        if not system:
            return self.user_rc_path
        return None

    def load_rc(self, path=None, prefix=None, system=False):
        """
        Load the raw conda configuration file using pyyaml.

        Depending on path or specific environment prefix and system that
        config file will be returned. If invalid or inexistent file, then an
        empty dictionary is returned.

        Parameters
        ----------
        path : str
            Path to conda configuration file.
        prefix : str
            Prefix path, to retrieve the specific prefix configuration file.
        system : bool
            Retrieve the system configuration file.
        """
        config_path = self.generate_config_path(path, prefix, system)

        data = {}
        if config_path and os.path.isfile(config_path):
            with open(config_path) as f:
                data = yaml.full_load(f)

        return data

    def load_rc_plain(self, path=None, prefix=None, system=False):
        """
        Loads the raw conda configuration file and returns in a 'plain' way
        i.e. 'as is' stored in the file.

        Parameters
        ----------
        path : str
            Path to conda configuration file.
        prefix : str
            Prefix path, to retrieve the specific prefix configuration file.
        system : bool
            Retrieve the system configuration file.
        """
        config_path = self.generate_config_path(path, prefix, system)

        if config_path and os.path.isfile(config_path):
            with open(config_path) as f:
                return f.read()

        return ''

    def save_rc(self, data, path=None, prefix=None, system=False):
        """
        Saves the yaml data into the .condarc file.

        Parameters
        ----------
        path : str
            Path to conda configuration file.
        prefix : str
            Prefix path, to retrieve the specific prefix configuration file.
        system : bool
            Retrieve the system configuration file.
        """
        config_path = self.generate_config_path(path, prefix, system)

        if config_path and os.path.isfile(config_path):
            with open(config_path, 'w') as f:
                yaml.dump(data, f)

        logger.debug('.condarc file was updated.')

    def save_rc_plain(self, data, path=None, prefix=None, system=False):
        """
        Saves the 'plain' yaml data into the .condarc file.

        Parameters
        ----------
        path : str
            Path to conda configuration file.
        prefix : str
            Prefix path, to retrieve the specific prefix configuration file.
        system : bool
            Retrieve the system configuration file.
        """
        config_path = self.generate_config_path(path, prefix, system)

        if config_path and os.path.isfile(config_path):
            with open(config_path, 'w') as f:
                f.write(data)

    # --- Additional methods
    # -------------------------------------------------------------------------
    def dependencies(self, name=None, prefix=None, pkgs=None, channels=None, dep=True):
        """Get dependenciy list for packages to be installed in an env."""
        if not pkgs or not isinstance(pkgs, (list, tuple)):
            raise TypeError('must specify a list of one or more packages to install into existing environment')

        cmd_list = ['install', '--dry-run', '--json', '--force-pscheck']

        if not dep:
            cmd_list.extend(['--no-deps'])

        if name:
            cmd_list.extend(['--name', name])
        elif prefix:
            cmd_list.extend(['--prefix', prefix])

        cmd_list.extend(pkgs)

        return self._call_and_parse(cmd_list)

    def environment_exists(self, name=None, prefix=None):
        """Check if an environment exists by 'name' or by 'prefix'.

        If query is by 'name' only the default conda environments directory is
        searched.
        """
        if name and prefix or (name is None and prefix is None):
            raise TypeError("Exactly one of 'name' or 'prefix' is required.")

        if name:
            prefix = self.get_prefix_envname(name)

        if prefix is None:
            prefix = self.ROOT_PREFIX

        return os.path.isdir(os.path.join(prefix, 'conda-meta'))

    def clear_lock(self):
        """Clean any conda lock in the system."""
        cmd_list = ['clean', '--lock', '--json']
        return self._call_and_parse(cmd_list)

    def package_version(self, prefix=None, name=None, pkg=None, build=False):
        """Get installed package version in a given env."""
        package_versions = {}

        if name and prefix:
            raise TypeError("Exactly one of 'name' or 'prefix' is required.")

        if name:
            prefix = self.get_prefix_envname(name)

        if self.environment_exists(prefix=prefix):

            for package in self.linked(prefix):
                if pkg in package:
                    n, v, b = split_canonical_name(package)
                    if build:
                        package_versions[n] = f'{v}={b}'
                    else:
                        package_versions[n] = v

        return package_versions.get(pkg)

    @staticmethod
    def get_platform():
        """Get platform of current system (system and bitness)."""
        _sys_map = {
            'linux2': 'linux',
            'linux': 'linux',
            'darwin': 'osx',
            'win32': 'win',
            'openbsd5': 'openbsd',
        }

        non_x86_linux_machines = {'armv6l', 'armv7l', 'ppc64le'}
        sys_platform = _sys_map.get(sys.platform, 'unknown')
        bits = 8 * tuple.__itemsize__

        arch_name = platform.machine()
        if (sys_platform == 'linux') and (arch_name in non_x86_linux_machines):
            subdir = f'linux-{arch_name}'
        else:
            subdir = f'{sys_platform}-{bits}'

        return subdir

    def load_proxy_config(self, path=None, system=None):
        """Load the proxy configuration."""
        config = self.load_rc(path=path, system=system)

        proxy_servers = {}
        HTTP_PROXY = os.environ.get('HTTP_PROXY')
        HTTPS_PROXY = os.environ.get('HTTPS_PROXY')

        if HTTP_PROXY:
            proxy_servers['http'] = HTTP_PROXY

        if HTTPS_PROXY:
            proxy_servers['https'] = HTTPS_PROXY

        proxy_servers_conf = config.get('proxy_servers', {})
        proxy_servers.update(proxy_servers_conf)

        return proxy_servers

    # --- Pip commands
    # -------------------------------------------------------------------------
    def _call_pip(self, name=None, prefix=None, extra_args=None, callback=None):
        """Call pip in QProcess worker."""
        cmd_list = self._pip_cmd(name=name, prefix=prefix)
        cmd_list.extend(extra_args)

        process_worker = ProcessWorker(cmd_list, pip=True, callback=callback)
        process_worker.sig_finished.connect(self._start)
        self._queue.append(process_worker)
        self._start()

        return process_worker

    def _pip_cmd(self, name=None, prefix=None):
        """Get pip location based on environment `name` or `prefix`."""
        if (name and prefix) or not (name or prefix):
            raise TypeError('conda pip: exactly one of \'name\' or \'prefix\' required.')

        if name and self.environment_exists(name=name):
            prefix = self.get_prefix_envname(name)

        python = get_pyexec(prefix)
        pip = get_pyscript(prefix, 'pip')

        cmd_list = [python, pip]

        return cmd_list

    def pip_list(self, name=None, prefix=None):
        """Get list of pip installed packages."""
        if (name and prefix) or not (name or prefix):
            raise TypeError('conda pip: exactly one of \'name\' or \'prefix\' required.')

        if name:
            prefix = self.get_prefix_envname(name)

        pyth_command = get_pyexec(prefix)

        try:
            cmd = 'import site; print([_ for _ in site.getsitepackages() if _.endswith(\'site-packages\')][0])'
            sp_dir = subprocess.check_output(   # nosec
                [pyth_command, '-E', '-c', cmd],
                creationflags=subprocess_utils.CREATE_NO_WINDOW,
            )
            sp_dir = sp_dir.strip()
            if hasattr(sp_dir, 'decode'):
                sp_dir = sp_dir.decode()
            cmd_list = [pyth_command, '-m', 'pip', 'list', '--format=json', '--pre', f'--path={sp_dir}']
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            cmd_list = ['echo', ' ']
        process_worker = ProcessWorker(
            cmd_list,
            pip=True,
            parse=True,
            callback=self._pip_list,
            extra_kwargs={'prefix': prefix},
        )
        process_worker.sig_finished.connect(self._start)
        self._queue.append(process_worker)
        self._start()

        return process_worker

    def _pip_list(self, stdout, stderr, prefix=None):
        """Callback for `pip_list`."""
        result = stdout
        linked = self.linked(prefix)
        pip_only = []
        linked_names = [split_canonical_name(link)[0] for link in linked]

        for pkg in result:
            name = split_canonical_name(pkg)[0]
            if name not in linked_names:
                pip_only.append(pkg)
            # NOTE: NEED A MORE ROBUST WAY!
            #            if '<pip>' in line and '#' not in line:
            #                temp = line.split()[:-1] + ['pip']
            #                temp = '-'.join(temp)
            #                if '-(' in temp:
            #                    start = temp.find('-(')
            #                    end = temp.find(')')
            #                    substring = temp[start:end+1]
            #                    temp = temp.replace(substring, '')
            #                result.append(temp)

        return pip_only

    def pip_remove(self, name=None, prefix=None, pkgs=None):
        """Remove a pip package in given environment by `name` or `prefix`."""
        if isinstance(pkgs, (list, tuple)):
            pkg = ' '.join(pkgs)
        else:
            pkg = pkgs

        extra_args = ['uninstall', '--yes', pkg]

        return self._call_pip(name=name, prefix=prefix, extra_args=extra_args)

    def _repodata_roots(self, pkgs_dirs: typing.Optional[typing.Sequence[str]] = None) -> typing.List[str]:
        """"""
        path: str
        for path in (pkgs_dirs or self.pkgs_dirs):
            try:
                open(os.path.join(path, 'urls.txt'), 'ab').close()  # pylint: disable=consider-using-with
            except OSError:
                pass
            else:
                return [os.path.join(path, 'cache')]
        return []

    def get_repodata(
            self,
            channels: typing.Optional[typing.Iterable[str]] = None,
            pkgs_dirs: typing.Optional[typing.Sequence[str]] = None,
    ) -> typing.Mapping[str, 'repodata.RepoData']:
        """Collect repodata from conda cache."""
        return repodata.REPO_CACHE.collect(
            directories=self._repodata_roots(pkgs_dirs=pkgs_dirs),
            channels=channels,
        )

    def get_repodata_modification_time(
            self,
            channels: typing.Optional[typing.Iterable[str]] = None,
            pkgs_dirs: typing.Optional[typing.Sequence[str]] = None,
    ) -> float:
        """Detect when repodata cache was updates last time."""
        return repodata.REPO_CACHE.modification_time(
            directories=self._repodata_roots(pkgs_dirs=pkgs_dirs),
            channels=channels,
        )


def CondaAPI():
    """Conda non blocking api."""
    global CONDA_API  # pylint: disable=global-statement

    if CONDA_API is None:
        CONDA_API = _CondaAPI()

    return CONDA_API
