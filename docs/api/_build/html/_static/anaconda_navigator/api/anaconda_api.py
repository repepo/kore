# -*- coding: utf-8 -*-

# pylint: disable=broad-except,invalid-name,too-many-lines,unspecified-encoding,unused-argument

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""API for using the api (anaconda-client, downloads and conda)."""

import itertools
from collections import OrderedDict
import bz2
import html
import json
import os
import re
import shutil
import sys
import typing

from qtpy.QtCore import QObject, Signal  # pylint: disable=no-name-in-module
import requests

from anaconda_navigator.api import external_apps
from anaconda_navigator.api.client_api import ClientAPI
from anaconda_navigator.api.conda_api import CondaAPI
from anaconda_navigator.api.download_api import DownloadAPI
from anaconda_navigator.api.process import WorkerManager
from anaconda_navigator.api.team_edition_api import TeamEditionAPI
from anaconda_navigator.api.utils import is_internet_available, split_canonical_name
from anaconda_navigator.config import CONF, LAUNCH_SCRIPTS_PATH, METADATA_PATH, WIN, AnacondaBrand
from anaconda_navigator.static import content, images
from anaconda_navigator.utils import basics
from anaconda_navigator.utils import constants as C, get_domain_from_api_url
from anaconda_navigator.utils import url_utils
from anaconda_navigator.utils.logs import logger
from anaconda_navigator.utils.misc import path_is_writable
from anaconda_navigator.utils.py3compat import is_binary_string
from . import types as api_types

if typing.TYPE_CHECKING:
    from anaconda_navigator.api.conda_api import ProcessWorker


class _AnacondaAPI(QObject):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """
    Anaconda Manager API.

    This class contains all methods from the different apis and acts as a controller for the main actions Navigator
    needs to execute.
    """
    sig_api_health = Signal(object)
    sig_metadata_updated = Signal(object)  # metadata_dictionary
    sig_repodata_loaded = Signal(object, object)  # packages, apps

    sig_repodata_updated = Signal(object)
    sig_repodata_errored = Signal()
    sig_error = Signal()

    def __init__(self):
        """Anaconda Manager API process worker."""
        super().__init__()

        # API's
        self._conda_api = CondaAPI()
        self._client_api = ClientAPI()
        self._download_api = DownloadAPI()
        self._process_api = WorkerManager()
        self.ROOT_PREFIX = self._conda_api.ROOT_PREFIX
        self.CONDA_PREFIX = self._conda_api.CONDA_PREFIX
        self._metadata = {}

        # Variables
        self._data_directory = None
        # Expose some methods for convenient access. Methods return a worker
        self.conda_dependencies = self._conda_api.dependencies
        self.conda_remove = self._conda_api.remove
        self.conda_terminate = self._conda_api.terminate_all_processes
        self.conda_config_add = self._conda_api.config_add
        self.conda_config_set = self._conda_api.config_set
        self.conda_config_remove = self._conda_api.config_remove
        self.download = self._download_api.download
        self.download_is_valid_url = self._download_api.is_valid_url
        _get_api_info = self._download_api.get_api_info
        _get_api_url = self._client_api.get_api_url
        self.download_is_valid_api_url = self._download_api.is_valid_api_url
        self.download_get_api_info = lambda: _get_api_info(_get_api_url())
        self.download_is_valid_channel = self._download_api.is_valid_channel
        self.download_terminate = self._download_api.terminate

        # No workers are returned for these methods
        self.conda_clear_lock = self._conda_api.clear_lock
        self.conda_environment_exists = self._conda_api.environment_exists
        self.conda_get_envs = self._conda_api.get_envs
        self.conda_linked = self._conda_api.linked
        self.conda_linked_apps_info = self._conda_api.linked_apps_info
        self.conda_get_prefix_envname = self._conda_api.get_prefix_envname
        self.conda_package_version = self._conda_api.package_version
        self.conda_platform = self._conda_api.get_platform
        self.conda_load_proxy_config = self._conda_api.load_proxy_config

        # These client methods return a worker
        self.client_login = self._client_api.login
        self.client_logout = self._client_api.logout
        self.client_user = self._client_api.user
        self.client_get_api_url = self._client_api.get_api_url
        self.client_set_api_url = self._client_api.set_api_url
        self.client_get_ssl = self._client_api.get_ssl
        self.client_set_ssl = self._client_api.set_ssl
        self.client_domain = self._client_api.domain
        self.client_reload = self._client_api.reload_client

    # --- Public API
    # -------------------------------------------------------------------------
    def set_data_directory(self, data_directory):
        """Set the directory where metadata is stored."""
        self._data_directory = data_directory

    # --- Client
    # -------------------------------------------------------------------------
    @staticmethod
    def is_offline():  # pylint: disable=missing-function-docstring
        return not is_internet_available()

    def login(self, username, password, verify_ssl=None):
        """
        Login to anaconda cloud via the anaconda-client API.

        This method does not use workers.
        """
        logger.debug('Login attempt with username `%s`.', username)
        return self._client_api.login(username, password, 'Anaconda Navigator', '', verify_ssl=verify_ssl)

    def logout(self):
        """
        Logout from anaconda cloud via the anaconda-client API.

        This method does not use workers.
        """
        logger.debug('Logout was requested!')
        return self._client_api.logout()

    def is_logged_in(self):
        """Check if an user is logged in."""
        return bool(self._client_api.user())

    def api_urls(self):
        """Get all the api urls for the current api url."""
        api_url = self._client_api.get_api_info_url()

        def _config(worker, output, error):
            base_worker = worker
            proxy_servers = output.get('proxy_servers', {})
            verify = output.get('ssl_verify', True)
            logger.debug('Requesting api info for `%s`.', api_url)
            worker = self._client_api.get_api_info(
                api_url,
                proxy_servers=proxy_servers,
                verify=verify,
            )
            worker.base_worker = base_worker
            worker.sig_finished.connect(_api_info)

        def _api_info(worker, output, error):
            base_worker = worker.base_worker
            base_worker.sig_chain_finished.emit(base_worker, output, error)

        worker = self._conda_api.config_show()
        worker.sig_finished.connect(_config)
        return worker

    # --- Conda
    # -------------------------------------------------------------------------
    @staticmethod
    def _process_unsafe_channels(channels, unsafe_channels):  # pylint: disable=too-many-locals
        """
        Fix channels with tokens so that we can correctly process conda cache.

        From this:
            - 'https://conda.anaconda.org/t/<TOKEN>/repo/goanpeca/<SUBDIR>'
        to this:
            - 'https://conda.anaconda.org/t/<ACTUAL-VALUE>/repo/goanpeca/<SUBDIR>'

        And from this:
            - 'https://conda.anaconda.org/repo/t/<TOKEN>/goanpeca/<SUBDIR>'
        to this:
            - 'https://conda.anaconda.org/t/<ACTUAL-VALUE>/repo/goanpeca/<SUBDIR>'
        """
        TOKEN_START_MARKS = ('t/', '/t/')
        TOKEN_START_MARKS_REPO = ('/repo/t/', )
        TOKEN_VALUE_MARK = '<TOKEN>'  # nosec

        token_channels = OrderedDict()
        for ch in unsafe_channels:
            for token_start_mark in TOKEN_START_MARKS:
                if token_start_mark in ch:
                    start, token_plus_user_and_system = ch.split(token_start_mark)
                    start = start + token_start_mark
                    parts = token_plus_user_and_system.split('/')
                    token = parts[0]
                    end = '/'.join([''] + parts[1:])
                    token_channels[start + TOKEN_VALUE_MARK + end] = token

            for token_start_mark in TOKEN_START_MARKS_REPO:
                if token_start_mark in ch:
                    start, token_plus_user_and_system = ch.split(token_start_mark)
                    parts = token_plus_user_and_system.split('/')
                    token = parts[0]
                    end = '/'.join(('repo', parts[1]))
                    start = '/'.join((start, 't'))
                    concat_channel = f'{start}/{TOKEN_VALUE_MARK}/{end}'
                    token_channels[concat_channel] = token

        new_channels = []
        for ch in channels:
            for token_start_mark in TOKEN_START_MARKS:
                if token_start_mark in ch:
                    for uch, token in token_channels.items():
                        if uch in ch:
                            ch = ch.replace(TOKEN_VALUE_MARK, token)
            new_channels.append(ch)

        return new_channels

    def conda_data(self, prefix=None):
        """
        Return all the conda data needed to make the application work.

        If prefix is None, the root prefix is used.
        """

        # On startup this should be loaded once
        if not self._metadata:
            self.load_bundled_metadata()

        def _load_unsafe_channels(base_worker, info, error):
            """"""
            new_worker = self._conda_api.info(prefix=prefix)
            new_worker.sig_finished.connect(_conda_info_processed)
            new_worker.unsafe_channels = info['channels']
            new_worker.base_worker = base_worker

        def _conda_info_processed(worker, info, error):
            base_worker = worker.base_worker
            processed_info = self._process_conda_info(info)
            # info = processed_info
            base_worker.info = info
            base_worker.processed_info = processed_info

            condarc = self._conda_api.load_rc()
            if condarc:
                rc_default_channels = condarc.get('default_channels', [])
                worker.unsafe_channels.extend(rc_default_channels)

            channels = self._process_unsafe_channels(info['channels'], worker.unsafe_channels)
            prefix = info['default_prefix']
            python_version = self._conda_api.package_version(pkg='python', prefix=prefix)
            pkgs_dirs = info['pkgs_dirs']
            logger.debug('Loading repodata for channels `%s` and package dirs `%s`', channels, pkgs_dirs)
            repodata = self._conda_api.get_repodata(channels=channels, pkgs_dirs=pkgs_dirs)

            if repodata:
                logger.debug('Extracting packages and apps from repodata.')
                new_worker = self._client_api.load_repodata(
                    repodata=repodata, metadata=self._metadata, python_version=python_version
                )
                new_worker.base_worker = base_worker
                new_worker.sig_finished.connect(_load_repo_data)
            else:
                # Force a refresh of the cache due to empty repodata
                new_worker = self._conda_api.search('conda', prefix=prefix)
                new_worker.base_worker = base_worker
                new_worker.channels = channels
                new_worker.pkgs_dirs = pkgs_dirs
                new_worker.python_version = python_version
                new_worker.sig_finished.connect(_get_repodata)

        def _get_repodata(worker, output, error):
            logger.debug('Loading repodata for channels `%s` and package dirs `%s`', worker.channels, worker.pkgs_dirs)
            repodata = self._conda_api.get_repodata(channels=worker.channels, pkgs_dirs=worker.pkgs_dirs)
            logger.debug('Extracting packages and apps from repodata.')
            new_worker = self._client_api.load_repodata(
                repodata=repodata, metadata=self._metadata, python_version=worker.python_version
            )
            new_worker.base_worker = worker.base_worker
            new_worker.sig_finished.connect(_load_repo_data)

        def _load_repo_data(worker, output, error):
            base_worker = worker.base_worker
            packages, applications = output
            new_output = {
                'info': base_worker.info,
                'processed_info': base_worker.processed_info,
                'packages': packages,
                'applications': applications,
            }
            # logger.debug('Processed conda data:\n%s', new_output)
            base_worker.sig_chain_finished.emit(base_worker, new_output, error)

        worker = self._conda_api.info(prefix=prefix, unsafe_channels=True)
        worker.sig_finished.connect(_load_unsafe_channels)
        return worker

    def conda_info(self, prefix=None):
        """
        Return the processed conda info for a given prefix.

        If prefix is None, the root prefix is used.
        """

        def _conda_info_processed(worker, info, error):
            processed_info = self._process_conda_info(info)
            worker.sig_chain_finished.emit(worker, processed_info, error)

        worker = self._conda_api.info(prefix=prefix)
        worker.sig_finished.connect(_conda_info_processed)
        return worker

    def conda_config(self, prefix=None):
        """Show config for a given prefix."""

        def _config(worker, output, error):
            worker.sig_chain_finished.emit(worker, {'config': output}, error)
        return self._create_worker(prefix=prefix, connected_method=_config)

    def conda_config_sources(self, prefix=None):
        """Show config sources for a given prefix."""

        def _config_sources(worker, output, error):
            worker.sig_chain_finished.emit(worker, {'config_sources': output}, error)
        return self._create_worker(prefix=prefix, connected_method=_config_sources)

    def conda_config_and_sources(self, prefix=None):
        """Show config and config sources for a given prefix."""

        def _config_sources(worker, output, error):
            base_worker = worker
            worker = self._conda_api.config_show(prefix=prefix)
            worker.config_sources = output
            worker.base_worker = base_worker
            worker.sig_finished.connect(_config)

        def _config(worker, output, error):
            base_worker = worker.base_worker
            new_output = {
                'config': output,
                'config_sources': worker.config_sources,
            }
            base_worker.sig_chain_finished.emit(base_worker, new_output, error)

        return self._create_worker(prefix=prefix, connected_method=_config_sources)

    def _create_worker(self, prefix: str, connected_method: typing.Callable) -> 'ProcessWorker':
        worker = self._conda_api.config_show_sources(prefix=prefix)
        worker.sig_finished.connect(connected_method)
        return worker

    @staticmethod
    def _process_conda_info(info):
        """Process conda info output and add some extra keys."""
        processed_info = info.copy()

        # Add a key for writable environment directories
        envs_dirs_writable = []
        for env_dir in info['envs_dirs']:
            if path_is_writable(env_dir):
                envs_dirs_writable.append(env_dir)
        processed_info['__envs_dirs_writable'] = envs_dirs_writable

        # Add a key for writable environment directories
        pkgs_dirs_writable = []
        for pkg_dir in info.get('pkgs_dirs'):
            if path_is_writable(pkg_dir):
                pkgs_dirs_writable.append(pkg_dir)
        processed_info['__pkgs_dirs_writable'] = pkgs_dirs_writable

        # Add a key for all environments
        root_prefix = info.get('root_prefix')
        environments = OrderedDict()
        environments[root_prefix] = 'base (root)'  # Ensure order
        envs = info.get('envs')
        envs_names = [os.path.basename(env) for env in envs]
        for env_name, env_prefix in sorted(zip(envs_names, envs)):
            if WIN:
                # See: https://github.com/ContinuumIO/navigator/issues/1496
                env_prefix = env_prefix[0].upper() + env_prefix[1:]

            environments[env_prefix] = env_name

        # Since conda 4.4.x the root environment is also listed, so we
        # "patch" the name of the env after processing all other envs
        environments[root_prefix] = 'base (root)'
        processed_info['__environments'] = _AnacondaAPI.filter_environments(environments)

        return processed_info

    @staticmethod
    def filter_environments(envs):
        """
        Removes all environments which names starts with underscore.

        :param OrderedDict envs: List of environments which will be filtered.

        :return OrderedDict: Filtered environments.
        """
        filtered_envs = OrderedDict()

        for key, env in envs.items():
            if not env.startswith('_'):
                filtered_envs[key] = env

        return filtered_envs

    def process_packages(self, packages, prefix=None, blacklist=()):
        """Process packages data and metadata to row format for table model."""

        def _call_list_prefix(base_worker, output, error):
            worker = self._conda_api.list(prefix=prefix)
            worker.base_output = output
            worker.base_worker = base_worker
            worker.sig_finished.connect(_pip_data_ready)

        def _pip_data_ready(worker, output, error):
            pip_list_data = worker.base_output

            base_worker = worker.base_worker
            clean_packages = base_worker.packages  # Blacklisted removed!

            if error:
                logger.error(error)

            pip_packages = pip_list_data or []

            # Get linked data
            linked = self._conda_api.linked(prefix=prefix)
            channel_urls = set(package['base_url'] for package in output if package['platform'] != 'pypi')

            platforms: typing.Tuple[str, ...] = (self._conda_api.get_platform(), 'noarch')
            metadata_channels: typing.List[str] = list({
                url_utils.join(base_url, platform)
                for base_url in channel_urls
                for platform in platforms
            })

            meta_repodata = self._conda_api.get_repodata(metadata_channels)
            packages, apps = self._client_api._load_repodata(  # pylint: disable=protected-access
                meta_repodata, self._metadata,
            )
            packages.update(apps)
            metadata = packages

            worker = self._client_api.prepare_model_data(clean_packages, linked, pip_packages, metadata)
            worker.base_worker = base_worker
            worker.sig_finished.connect(_model_data_ready)

        def _model_data_ready(worker, output, error):
            base_worker = worker.base_worker
            clean_packages = base_worker.packages
            data = output[:]

            # Remove blacklisted packages (Double check!)
            for package_name in blacklist:
                if package_name in clean_packages:
                    clean_packages.pop(package_name)

                row: int
                for row in reversed(range(len(data))):
                    if data[row][C.COL_NAME] == package_name:
                        data.pop(row)

            # Worker, Output, Error
            base_worker.sig_chain_finished.emit(base_worker, (clean_packages, data), error)

        # Remove blacklisted packages, copy to avoid mutating packages dict!
        # See: https://github.com/ContinuumIO/navigator/issues/1244
        clean_packages = packages.copy()
        for package_name in blacklist:
            if package_name in clean_packages:
                clean_packages.pop(package_name)

        # Get pip data
        worker = self._conda_api.pip_list(prefix=prefix)
        worker.packages = clean_packages
        worker.sig_finished.connect(_call_list_prefix)

        return worker

    def process_apps(  # pylint: disable=too-many-locals
            self,
            apps: typing.Mapping[api_types.ApplicationName, 'api_types.RawApplication'],
            prefix: typing.Optional[str] = None,
    ) -> typing.Dict[str, 'api_types.Application']:
        """Process app information."""
        if prefix is None:
            prefix = self.ROOT_PREFIX

        applications: typing.Dict[str, 'api_types.Application'] = {}
        collected_applications: external_apps.ApplicationCollection = external_apps.get_applications(
            configuration=CONF,
            process_api=self._process_api,
        )

        # These checks installed apps in the prefix
        linked_apps_info: typing.Dict[str, 'api_types.Application']
        linked_apps_info = self.conda_linked_apps_info(prefix, envs_cache=True)

        local_apps_info: typing.Dict[str, 'api_types.Application'] = self._conda_api.local_apps_info()

        missing_apps: typing.Mapping[api_types.ApplicationName, 'api_types.RawApplication'] = {
            key: value
            for key, value in self.conda_linked_apps_info(prefix).items()
            if key not in apps
        }

        app_name: api_types.ApplicationName
        app_data: 'api_types.RawApplication'
        for app_name, app_data in itertools.chain(apps.items(), missing_apps.items()):
            versions: typing.Sequence[str] = app_data.get('versions', [])
            if not versions:
                continue

            # collecting patches

            application_patch: external_apps.AppPatch
            application_patch = external_apps.MANUAL_PATCHES.get(app_name, external_apps.EMPTY_PATCH)

            extra_patch: external_apps.AppPatch
            extra_patch = collected_applications.app_patches.get(app_name, external_apps.EMPTY_PATCH)
            application_patch = extra_patch.apply_to(application_patch)

            if not application_patch.is_available:
                continue

            # collecting base information

            linked_app_info: 'api_types.Application' = linked_apps_info.get(app_name, {})
            local_app_info: 'api_types.Application' = local_apps_info.get(app_name, {})

            description: str = basics.coalesce(
                linked_app_info.get('description', None),
                local_apps_info.get(app_name, {}).get('description', None),
                '',
            )

            image_path: str = basics.coalesce(
                linked_app_info.get('image_path', None),
                local_app_info.get('image_path', None),
                images.ANACONDA_ICON_256_PATH,
            )

            latest_version: api_types.Version = app_data.get('latest_version') or versions[-1]

            installed_version: typing.Optional[api_types.Version]
            installed_version = self.conda_package_version(prefix=prefix, pkg=app_name, build=False)

            app_entries: typing.Mapping[api_types.Version, str] = app_data.get('app_entry', {})
            command: typing.Optional[str] = app_entries.get(latest_version, '')
            if installed_version:
                command = app_entries.get(installed_version, command)
            if not command:
                continue
            command = re.sub(r'(?i:ipython\s+(?=notebook|qtconsole))', 'jupyter-', command, 1)

            applications[app_name] = {
                'app_type': C.AppType.CONDA,
                'command': command,
                'description': description,
                'display_name': app_name,
                'image_path': image_path,
                'installed': bool(installed_version),
                'name': app_name,
                'non_conda': False,
                'rank': 0,
                'version': installed_version or latest_version,
                'versions': versions,
            }
            application_patch.apply_to(applications[app_name])

        web_app: 'external_apps.BaseWebApp'
        for web_app in collected_applications.web_apps.values():
            if not web_app.is_available:
                continue
            applications[web_app.app_name] = web_app.tile_definition

        app: 'external_apps.BaseInstallableApp'
        for app in collected_applications.installable_apps.values():
            if not app.is_available:
                continue
            if app.non_conda and (not app.executable) and (not app.is_installation_enabled):
                continue
            applications[app.app_name] = app.tile_definition

        return applications

    # --- Conda environments
    # -------------------------------------------------------------------------
    def load_bundled_metadata(self):
        """Load bundled metadata."""
        comp_meta_filepath = content.BUNDLE_METADATA_COMP_PATH
        conf_meta_filepath = content.CONF_METADATA_PATH
        conf_meta_folder = METADATA_PATH

        if os.path.exists(conf_meta_filepath):
            try:
                with open(conf_meta_filepath, 'r') as json_file:
                    self._metadata = json.load(json_file).get('packages', {})
                logger.debug('Metadata was loaded from `%s`', conf_meta_filepath)
            except Exception as e:
                logger.exception(e)
                self._metadata = {}
                logger.debug('Metadata is empty!')

            finally:
                return  # pylint: disable=lost-exception

        try:
            os.makedirs(conf_meta_folder, exist_ok=True)
        except OSError:
            pass

        binary_data = None
        if comp_meta_filepath and os.path.isfile(comp_meta_filepath):
            with open(comp_meta_filepath, 'rb') as f:
                binary_data = f.read()

        if binary_data:
            try:
                data = bz2.decompress(binary_data)
                with open(conf_meta_filepath, 'wb') as f:
                    f.write(data)

                if is_binary_string(data):
                    data = data.decode()

                self._metadata = json.loads(data).get('packages', {})
                logger.debug('Metadata was loaded from `%s`', comp_meta_filepath)
            except Exception as e:
                logger.error(e)
                self._metadata = {}
                logger.debug('Metadata is empty!')

    def update_index_and_metadata(self, prefix=None):
        """
        Update the metadata available for packages in repo.anaconda.com.

        Returns a download worker with chained finish signal.
        """
        def _metadata_updated(worker, path, error):
            """Callback for update_metadata."""
            base_worker = worker
            if path and os.path.isfile(path):
                with open(path, 'r') as f:
                    data = f.read()
            try:
                self._metadata = json.loads(data).get('packages', {})
                logger.debug('Metadata was loaded from `%s`', path)
            except Exception as e:
                logger.exception(e)
                self._metadata = {}
                logger.debug('Metadata is empty!')

            worker = self._conda_api.search('conda', prefix=prefix)
            worker.base_worker = base_worker
            worker.sig_finished.connect(_index_updated)

        def _index_updated(worker, output, error):
            base_worker = worker.base_worker
            base_worker.sig_chain_finished.emit(base_worker, None, None)

        # NOTE: there needs to be an uniform way to query the metadata for both repo and anaconda.org
        if self._data_directory is None:
            raise Exception('Need to call `api.set_data_directory` first.')   # pylint: disable=broad-exception-raised

        metadata_url = 'https://repo.anaconda.com/pkgs/main/channeldata.json'
        filepath = content.CONF_METADATA_PATH
        worker = self.download(metadata_url, filepath)
        worker.action = C.ACTION_SEARCH
        worker.prefix = prefix
        worker.old_prefix = prefix
        worker.sig_finished.connect(_metadata_updated)
        logger.debug('Downloading metadata from `%s` into `%s`.', metadata_url, filepath)
        return worker

    def create_environment(
            self,
            prefix,
            packages=('python',),
            no_default_python=False,
    ):
        """Create environment and install `packages`."""
        worker = self._conda_api.create(
            prefix=prefix,
            pkgs=packages,
            no_default_python=no_default_python,
            offline=not is_internet_available(),
        )
        worker.action = C.ACTION_CREATE
        worker.action_msg = f'Creating environment <b>{prefix}</b>'
        worker.prefix = prefix
        worker.name = self._conda_api.get_name_envprefix(prefix)
        logger.debug('Creating environment `%s` with following packages: %s', prefix, packages)
        return worker

    def clone_environment(self, clone_from_prefix, prefix):
        """Clone environment located at `clone` (prefix) into name."""
        worker = self._conda_api.clone_environment(
            clone_from_prefix,
            prefix=prefix,
            offline=not is_internet_available()
        )
        worker.action = C.ACTION_CLONE
        clone_from_name = self._conda_api.get_name_envprefix(clone_from_prefix)
        worker.action_msg = f'Cloning from environment <b>{clone_from_name}</b> into <b>{prefix}</b>'
        worker.prefix = prefix
        worker.name = self._conda_api.get_name_envprefix(prefix)
        worker.clone = clone_from_prefix
        logger.debug('Cloning from environment `%s` into `%s`.', clone_from_prefix, prefix)
        return worker

    def export_environment(self, prefix, file):
        """Export environment, that exists in `prefix`, to the yaml `file`."""
        worker = self._conda_api.export_environment(file=file, prefix=prefix)
        worker.action = C.ACTION_EXPORT
        worker.action_msg = f'Backing up environment <b>{prefix}</b>'
        worker.prefix = prefix
        worker.name = self._conda_api.get_name_envprefix(prefix)
        worker.file = file

        logger.debug('Backing up environment `%s` into `%s`.', prefix, file)
        return worker

    def import_environment(self, prefix: str, file: str, validate_only: bool = False) -> 'ProcessWorker':
        """Import new environment on `prefix` with specified `file`."""
        worker = self._conda_api.create(
            prefix=prefix,
            file=file,
            offline=not is_internet_available(),
            dry_run=validate_only
        )
        worker.action = C.ACTION_IMPORT
        if validate_only:
            worker.action_msg = 'Validating environment'
            logger.debug('Validating environment `%s`.', prefix)
        else:
            worker.action_msg = f'Importing environment <b>{html.escape(prefix)}</b>'
            logger.debug('Importing environment `%s` from `%s`.', prefix, file)
        worker.prefix = prefix
        worker.name = self._conda_api.get_name_envprefix(prefix)
        worker.file = file
        return worker

    def remove_environment(self, prefix):
        """Remove environment `name`."""
        worker = self._conda_api.remove_environment(
            prefix=prefix,
            offline=not is_internet_available()
        )
        worker.action = C.ACTION_REMOVE_ENV
        worker.action_msg = f'Removing environment <b>{prefix}</b>'
        worker.prefix = prefix
        worker.name = self._conda_api.get_name_envprefix(prefix)
        logger.debug('Removing environment `%s`', prefix)

        # Remove scripts folder
        scripts_path = LAUNCH_SCRIPTS_PATH
        if prefix != self.ROOT_PREFIX:
            scripts_path = os.path.join(scripts_path, worker.name)
        try:
            shutil.rmtree(scripts_path)
            logger.debug('Scripts path `%s` was removed.', scripts_path)
        except OSError:
            pass

        return worker

    def install_packages(self, prefix, pkgs, dry_run=False, no_default_python=False):
        """Install `pkgs` in environment `prefix`."""
        worker = self._conda_api.install(
            prefix=prefix,
            pkgs=pkgs,
            dry_run=dry_run,
            no_default_python=no_default_python,
            offline=not is_internet_available(),
        )
        worker.action_msg = f'Installing packages on <b>{prefix}</b>'
        worker.action = C.ACTION_INSTALL
        worker.dry_run = dry_run
        worker.prefix = prefix
        worker.name = self._conda_api.get_name_envprefix(prefix)
        worker.pkgs = pkgs
        logger.debug(
            'Installing packages on prefix `%s`. (dry_run=%s) \n Packages to install: %s', prefix, dry_run, pkgs)
        return worker

    def update_packages(  # pylint: disable=too-many-arguments
        self,
        prefix,
        pkgs=None,
        dry_run=False,
        no_default_python=False,
        all_=False,
    ):
        """Update `pkgs` in environment `prefix`."""
        worker = self._conda_api.update(
            prefix=prefix,
            pkgs=pkgs,
            dry_run=dry_run,
            no_default_python=no_default_python,
            all_=all_,
            offline=not is_internet_available(),
        )
        worker.action_msg = f'Updating packages on <b>{prefix}</b>'
        worker.action = C.ACTION_UPDATE
        worker.dry_run = dry_run
        worker.prefix = prefix
        worker.name = self._conda_api.get_name_envprefix(prefix)
        worker.pkgs = pkgs

        logger.debug('Updating packages on prefix `%s`. (dry_run=%s) \n Packages to update: %s', prefix, dry_run, pkgs)
        return worker

    def remove_packages(self, prefix, pkgs, dry_run=False):
        """Remove `pkgs` from environment `prefix`."""
        worker = self._conda_api.remove(
            prefix=prefix,
            pkgs=pkgs,
            dry_run=dry_run,
            offline=not is_internet_available(),
        )
        worker.action_msg = f'Removing packages from <b>{prefix}</b>'
        worker.action = C.ACTION_REMOVE
        worker.prefix = prefix
        worker.name = self._conda_api.get_name_envprefix(prefix)
        worker.pkgs = pkgs
        logger.debug(
            'Removing packages from prefix `%s`. (dry_run=%s) \n Packages to remove: %s', prefix, dry_run, pkgs)
        return worker

    @staticmethod
    def check_navigator_dependencies(actions, prefix):  # pylint: disable=too-many-branches,too-many-locals
        """Check if navigator is affected by the operation on (base/root)."""

        # Check that the dependencies are not changing the current prefix
        # This allows running this check on any environment that navigator
        # is installed on, instead of hardcoding self.ROOT_PREFIX
        if prefix != sys.prefix:
            conflicts = False
        else:
            # Minimum requirements to disable downgrading
            conflicts = False
            if actions and isinstance(actions, list):
                actions = actions[0]

            if actions:
                unlinked = actions.get('UNLINK', [])

                try:
                    # Old conda json format
                    unlinked = {split_canonical_name(p)[0]: split_canonical_name(p) for p in unlinked}
                except AttributeError:
                    # New conda json format
                    unlinked = {
                        split_canonical_name(p['dist_name'])[0]:
                            split_canonical_name(p['dist_name']) for p in unlinked
                    }

                for pkg in unlinked:
                    if pkg == 'anaconda-navigator':
                        conflicts = True
                        break

        return conflicts

    @staticmethod
    def __modify_condarc(rc_data, default_channels=None, channels=None, channel_alias=None):
        """
            Replace `channels`, `default_channels`, `channel_alias` with a new or empty data
        """
        if channel_alias:
            rc_data['channel_alias'] = channel_alias
            logger.debug('.condarc file was updated with following `channel_alias`:\n%s', channel_alias)

        rc_data.setdefault('channels', [])
        if default_channels:
            rc_data['default_channels'] = default_channels
            rc_data['channels'] = ['defaults']
            logger.debug('.condarc file was updated with following `default_channels`:\n%s', default_channels)

        if channels:
            rc_data['channels'].extend(channels)
            logger.debug('.condarc file was updated with following `channels`:\n%s', channels)

        return rc_data

    def generate_rc_key(
            self,
            logged_brand: typing.Optional[str] = None,
            logged_api_url: typing.Optional[str] = None,
            user_id: typing.Optional[str] = None,
    ) -> typing.Optional[str]:
        """Generate unique identifier for current .condarc."""
        if logged_brand is None:
            logged_brand = CONF.get(CONF.DEFAULT_SECTION_NAME, 'logged_brand')
        if logged_api_url is None:
            logged_api_url = CONF.get(CONF.DEFAULT_SECTION_NAME, 'logged_api_url')
        if (not logged_brand) or (not logged_api_url):
            self._conda_api.rc_index.current = None
            return None

        if user_id is None:
            try:
                user_id = self.get_user_identifier()
            except requests.exceptions.RequestException:
                # ideally:
                # return self._conda_api.rc_index.current
                #
                # but for now it should be enough:
                self._conda_api.rc_index.current = None
                return None

        result: str = self._conda_api.rc_index.generate_config_index_key(logged_brand, logged_api_url, user_id)
        self._conda_api.rc_index.current = result

        logger.debug(
            'A new key `%s` was generated from following items: \nbrand: `%s` domain: `%s` user_id: `%s`',
            result,
            logged_brand,
            logged_api_url,
            user_id,
        )
        return result

    def __replace_condarc(
            self,
            rc_key: typing.Optional[str] = None,
            channel_alias: typing.Optional[str] = None,
    ) -> None:
        """
        Saves the copy of the .condarc data and replaces it with a modified snapshot.

        :param rc_key: Key to get snapshot for.
        :param channel_alias: Alias of the channel to integrate into restored snapshot.
        """
        rc_data = self._conda_api.load_rc()
        self._conda_api.rc_index.save_rc_copy(data=rc_data)

        if rc_key is not None:
            rc_data = self._conda_api.rc_index.load_rc_copy(rc_key)
            rc_data = self.__modify_condarc(rc_data, channel_alias=channel_alias)

            self._conda_api.rc_index.save_rc_copy(data=rc_data, rc_key=rc_key)
            self._conda_api.save_rc(rc_data)

    def restore_condarc(self, rc_key: typing.Optional[str] = None) -> None:
        """Load the data which was before user logged in."""
        if rc_key is not None:
            rc_data = self._conda_api.load_rc()
            self._conda_api.rc_index.save_rc_copy(data=rc_data, rc_key=rc_key)

        rc_data = self._conda_api.rc_index.load_rc_copy()
        self._conda_api.save_rc(data=rc_data)
        self._conda_api.rc_index.current = None

    def update_channels(self, default_channels=None, channels=None):  # pylint: disable=missing-function-docstring
        rc_data = self._conda_api.load_rc()
        rc_data = self.__modify_condarc(rc_data, default_channels, channels)
        self._conda_api.save_rc(rc_data)

    def create_login_data(self):
        """
        Creates the login data needed to interact with Anaconda Server instance.

        Updates .condarc data with channels which are accessed by authenticated user.
        Updates anaconda-navigator.ini file with Anaconda Server access token and token ID.
        """
        logged_brand: typing.Optional[str]
        logged_api_url: typing.Optional[str]
        logged_brand, logged_api_url = CONF.get_logged_data()

        rc_key: str = self.generate_rc_key(logged_brand=logged_brand, logged_api_url=logged_api_url)

        if logged_brand == AnacondaBrand.TEAM_EDITION:
            channel_alias = url_utils.join(logged_api_url, 'api/repo')
            self.__replace_condarc(rc_key=rc_key, channel_alias=channel_alias)

            jwt_token_data = json.loads(self._client_api.anaconda_client_api.load_token())
            if jwt_token_data:
                access_token = self._client_api.anaconda_client_api.create_access_token(jwt_token_data)
                CONF.set('main', 'anaconda_server_token', access_token['token'])
                CONF.set('main', 'anaconda_server_token_id', access_token['id'])
                logger.debug('%s access token was generated.', logged_brand)

        if logged_brand == AnacondaBrand.ENTERPRISE_EDITION:
            channel_alias = url_utils.join(get_domain_from_api_url(logged_api_url), 'conda')
            self.__replace_condarc(rc_key=rc_key, channel_alias=channel_alias)

    def remove_login_data(self):
        """
        Removes the login data needed to interact with Anaconda Server instance.

        Updates .condarc data by removing channels which were accessed by authenticated user.
        Updates anaconda-navigator.ini file with removing Anaconda Server access token and token ID.
        """
        logged_brand: typing.Optional[str]
        logged_api_url: typing.Optional[str]
        logged_brand, logged_api_url = CONF.get_logged_data()
        rc_key: str = self.generate_rc_key(logged_brand=logged_brand, logged_api_url=logged_api_url)

        if logged_brand == AnacondaBrand.TEAM_EDITION:

            access_token_id = CONF.get('main', 'anaconda_server_token_id')

            self.restore_condarc(rc_key=rc_key)
            try:
                self._client_api.anaconda_client_api.remove_access_token(access_token_id)
                logger.debug('%s access token was removed', logged_brand)
            except requests.exceptions.RequestException:
                pass

            CONF.set_logged_data()

        if logged_brand == AnacondaBrand.ENTERPRISE_EDITION:
            self.restore_condarc(rc_key=rc_key)

    def get_channels(self):
        """
        Returns the list of available channels.

        :return list[dict[str, mixed]]: The list with dictionaries with info about channels.
        """
        channels = []
        if isinstance(self._client_api.anaconda_client_api, TeamEditionAPI):
            channels = self._client_api.anaconda_client_api.get_channels()

        logger.debug('Current available channels are \n `%s`', channels)
        return channels

    def health_check(self):
        """
        Returns the list of available channels.

        :return list[dict[str, mixed]]: The list with dictionaries with info about channels.
        """
        if isinstance(self._client_api.anaconda_client_api, TeamEditionAPI):
            worker = self._process_api.create_python_worker(self._client_api.anaconda_client_api.ping)
            worker.sig_finished.connect(lambda _, healthy, error: self.sig_api_health.emit(healthy))
            worker.start()
            return

        self.sig_api_health.emit(True)

    def get_user_identifier(self):  # pylint: disable=missing-function-docstring
        if isinstance(self._client_api.anaconda_client_api, TeamEditionAPI):
            user_id = self._client_api.anaconda_client_api.get_user_id()
        else:
            user_id = self.client_user().get('login', '')

        logger.debug('Current user_id is `%s`', user_id)
        return user_id

    def client_reset_ssl(self) -> None:
        """Reset ssl preferences in clients to the current Navigator settings."""
        ssl_verification: bool = CONF.get('main', 'ssl_verification', True)
        ssl_certificate: typing.Optional[str] = CONF.get('main', 'ssl_certificate', None)
        if ssl_verification and ssl_certificate:
            self.client_set_ssl(ssl_certificate)
            logger.debug('SSL certificate was changed to `%s`.', ssl_certificate)
        else:
            logger.debug('SSL verification was changed to `%s`.', ssl_certificate)
            self.client_set_ssl(ssl_verification)


ANACONDA_API = None


def AnacondaAPI():
    """Manager API threaded worker."""
    global ANACONDA_API  # pylint: disable=global-statement

    if ANACONDA_API is None:
        ANACONDA_API = _AnacondaAPI()

    return ANACONDA_API
