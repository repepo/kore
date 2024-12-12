# -*- coding: utf-8 -*-

# pylint: disable=broad-except,invalid-name

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Worker threads for using the anaconda-client api."""

import itertools
import json
import logging
import time
import typing
from collections import deque

import binstar_client
import requests
from binstar_client.errors import Unauthorized
from qtpy.QtCore import QObject, QThread, QTimer, Signal  # pylint: disable=no-name-in-module
from requests.exceptions import SSLError

from anaconda_navigator.api.conda_api import CondaAPI
from anaconda_navigator.api.team_edition_api import TeamEditionAPI
from anaconda_navigator.api.utils import is_internet_available
from anaconda_navigator.config import CONF, AnacondaBrand
from anaconda_navigator.utils import anaconda_solvers
from anaconda_navigator.utils import constants as C
from anaconda_navigator.utils import notifications
from anaconda_navigator.utils import sort_versions
from anaconda_navigator.utils import url_utils
from anaconda_navigator.utils.logs import http_logger, logger
from anaconda_navigator.utils.py3compat import is_text_string, to_text_string
from . import utils as api_utils

if typing.TYPE_CHECKING:
    from binstar_client import Binstar


class ClientWorker(QObject):
    """Anaconda Client API process worker."""

    sig_chain_finished = Signal(object, object, object)
    sig_finished = Signal(object, object, object)

    def __init__(self, method, args, kwargs):
        """Anaconda Client API process worker."""
        super().__init__()
        self.method = method
        self.args = args
        self.kwargs = kwargs
        self._is_finished = False

    def is_finished(self):
        """Return whether or not the worker has finished running the task."""
        return self._is_finished

    def start(self):
        """Start the worker process."""
        error, output = None, None
        try:
            time.sleep(0.01)
            output = self.method(*self.args, **self.kwargs)
        except Exception as err:
            error = str(err)
            error = error.replace('(', '')
            error = error.replace(')', '')

        self.sig_finished.emit(self, output, error)
        self._is_finished = True


class Args:  # pylint: disable=too-few-public-methods
    """Dummy class to pass to anaconda client on token loading and removal."""


class _ClientAPI(QObject):  # pylint: disable=too-many-instance-attributes
    """Anaconda Client API wrapper."""

    DEFAULT_TIMEOUT = 6

    def __init__(self):
        """Anaconda Client API wrapper."""
        super().__init__()
        self._conda_api = CondaAPI()
        self._anaconda_client_api = None
        self._queue = deque()
        self._threads = []
        self._workers = []
        self._timer = QTimer()

        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._clean)

        # Setup
        CONF.set_logged_data()
        self.reload_client()

    @property
    def anaconda_client_api(self):  # pylint: disable=missing-function-docstring
        return self._anaconda_client_api

    def _clean(self):
        """Check for inactive workers and remove their references."""
        if self._workers:
            for w in self._workers:
                if w.is_finished():
                    self._workers.remove(w)

        if self._threads:
            for t in self._threads:
                if t.isFinished():
                    self._threads.remove(t)
        else:
            self._timer.stop()

    def _start(self):
        """Take avalaible worker from the queue and start it."""
        if len(self._queue) == 1:
            thread = self._queue.popleft()
            thread.start()
            self._timer.start()

    def _create_worker(self, method, *args, **kwargs):
        """Create a worker for this client to be run in a separate thread."""
        # NOTE: this might be heavy...
        thread = QThread()
        worker = ClientWorker(method, args, kwargs)
        worker.moveToThread(thread)
        worker.sig_finished.connect(self._start)
        worker.sig_finished.connect(thread.quit)
        thread.started.connect(worker.start)
        self._queue.append(thread)
        self._threads.append(thread)
        self._workers.append(worker)
        self._start()
        return worker

    # --- Callbacks
    # -------------------------------------------------------------------------
    @staticmethod
    def _load_repodata(repodata, metadata=None, python_version=None):  # pylint: disable=too-many-locals,unused-argument
        """
        Load all the available package information.

        See load_repadata for full documentation.
        """
        if metadata is None:
            metadata = {}

        all_packages = {}
        for repodata_value in repodata.values():
            for canonical_name, data in itertools.chain(
                    repodata_value.get('packages', {}).items(),
                    repodata_value.get('packages.conda', {}).items(),
            ):
                # Do not filter based on python version
                # if python_version and not is_dependency_met(data['depends'], python_version, 'python'):
                #     continue
                name, version, _ = tuple(canonical_name.rsplit('-', 2))

                if name not in all_packages:
                    all_packages[name] = {
                        'versions': set(),
                        'size': {},
                        'type': {},
                        'app_entry': {},
                        'app_type': {},
                    }
                elif name in metadata:
                    temp_data = all_packages[name]
                    temp_data['home'] = metadata[name].get('home', '')
                    temp_data['summary'] = metadata[name].get('summary', '')
                    temp_data['latest_version'] = metadata[name].get('version')
                    all_packages[name] = temp_data

                all_packages[name]['versions'].add(version)
                all_packages[name]['size'][version] = data.get('size', '')

                # Only the latest builds will have the correct metadata for
                # apps, so only store apps that have the app metadata
                if data.get('type'):
                    all_packages[name]['type'][version] = data.get('type')
                    all_packages[name]['app_entry'][version] = data.get('app_entry')
                    all_packages[name]['app_type'][version] = data.get('app_type')

        # Calculate the correct latest_version
        for package in all_packages.values():
            versions = tuple(sorted(package['versions'], reverse=True))
            package['latest_version'] = versions[0]

        all_apps = {}
        for name, package in all_packages.items():
            versions = sort_versions(list(package['versions']))
            package['versions'] = versions[:]

            # Has type in this case implies being an app
            if package.get('type'):
                all_apps[name] = package.copy()
                # Remove all versions that are not apps!
                all_apps[name]['versions'] = [
                    version
                    for version in all_apps[name]['versions']
                    if version in all_apps[name]['type']
                ]
        return all_packages, all_apps

    @staticmethod
    def _prepare_model_data(packages, linked, pip=None, metadata=None):  # pylint: disable=too-many-locals
        """Prepare model data for the packages table model."""
        pip = pip if pip else []
        data = []
        linked_packages = {}
        for canonical_name in linked:
            name, version, _ = tuple(canonical_name.rsplit('-', 2))
            linked_packages[name] = {'version': version}

        pip_packages = {}
        for canonical_name in pip:
            name, version, _ = tuple(canonical_name.rsplit('-', 2))
            pip_packages[name] = {'version': version}

        packages_names = sorted(
            list(set(list(linked_packages.keys()) + list(pip_packages.keys()) + list(packages.keys())), )
        )
        packages_metadata = metadata or {}
        for name in packages_names:
            p_data = packages.get(name) or packages_metadata.get(name, {})

            summary = p_data.get('summary') or ''
            url = p_data.get('home') or ''
            versions = p_data.get('versions') or []
            version = p_data.get('latest_version') or ''

            if name in pip_packages:
                type_ = C.PIP_PACKAGE
                version = pip_packages[name].get('version', '')
                status = C.INSTALLED
            elif name in linked_packages:
                type_ = C.CONDA_PACKAGE
                version = linked_packages[name].get('version', '')
                status = C.INSTALLED

                if version in versions:
                    vers = versions
                    upgradable = not version == vers[-1] and len(vers) != 1
                    downgradable = not version == vers[0] and len(vers) != 1

                    if upgradable and downgradable:
                        status = C.MIXGRADABLE
                    elif upgradable:
                        status = C.UPGRADABLE
                    elif downgradable:
                        status = C.DOWNGRADABLE
            else:
                type_ = C.CONDA_PACKAGE
                status = C.NOT_INSTALLED

            row = {
                C.COL_ACTION: C.ACTION_NONE,
                C.COL_PACKAGE_TYPE: type_,
                C.COL_NAME: name,
                C.COL_DESCRIPTION: summary.capitalize(),
                C.COL_VERSION: version,
                C.COL_STATUS: status,
                C.COL_URL: url,
                C.COL_ACTION_VERSION: None,
            }

            data.append(row)
        return data

    # --- Public API
    # -------------------------------------------------------------------------
    def reload_client(self) -> typing.Union['Binstar', TeamEditionAPI, None]:
        """
        Sets the client depending on the settings from the configuration file.
        """
        logged_brand: typing.Optional[str]
        logged_api_url: typing.Optional[str]
        anaconda_api_url: typing.Optional[str] = CONF.get('main', 'anaconda_api_url', None)
        logged_brand, logged_api_url = CONF.get_logged_data()

        if logged_brand in [AnacondaBrand.ANACONDA_ORG, AnacondaBrand.ENTERPRISE_EDITION]:
            return self._load_binstar_client(logged_api_url)

        if logged_brand == AnacondaBrand.TEAM_EDITION:
            return self._load_team_edition_client(logged_api_url)

        # Looks like there wasn't any action to login from the Navigator application.
        # Checking if there was a login action into Binstar client through the CLI.

        url = anaconda_api_url
        with anaconda_solvers.catch_and_notify():
            url = binstar_client.utils.get_config()['url']

        while True:  # while issues are fixed
            try:
                client = self._load_binstar_client(url)
                client.user()
                return client

            except (Unauthorized, SSLError, ValueError):
                # No users authorized through banister client.
                # Return Binstar client with default Anaconda API url.
                return self._load_binstar_client(anaconda_api_url)

            except requests.exceptions.ConnectionError:
                # No connection so we are using default client.
                return self._load_binstar_client(anaconda_api_url)

            except binstar_client.BinstarError as error:
                # Try to logout user without a validated email
                if (error.args[1:2] == (403,)) or (error.message == 'Email verification failed!'):
                    self.logout()
                    notifications.NOTIFICATION_QUEUE.push(
                        message=(
                            'Your email is not verified. Please verify it on anaconda.org and log into your account '
                            'again.'
                        ),
                        caption='Email not verified',
                        tags=('anaconda', 'email_verification'),
                    )
                    continue

                raise

    def _load_binstar_client(self, api_url):
        """
        Recreate the binstar client with new updated values.

        Notes:
        ------
        The Client needs to be restarted because on domain change it will not
        validate the user since it will check against the old domain, which
        was used to create the original client.

        See: https://github.com/ContinuumIO/navigator/issues/1325
        """
        config: typing.Any = {}
        with anaconda_solvers.catch_and_notify():
            config = binstar_client.utils.get_config()

        config['url'] = api_url
        for site in config['sites'].values():
            site['url'] = api_url

        with anaconda_solvers.catch_and_notify():
            binstar_client.utils.set_config(config)

        token = self.load_token()
        binstar = binstar_client.utils.get_server_api(
            token=token, site=None, cls=None, config=config, log_level=logging.NOTSET
        )
        self._anaconda_client_api = binstar
        return binstar

    def _load_team_edition_client(self, api_url):
        """
        Sets the '_anaconda_client_api' to user TeamEdition API instead of
        default Binstar client.
        """
        verify_ssl = self._conda_api.load_rc().get('ssl_verify', False)
        self._anaconda_client_api = TeamEditionAPI(api_url, verify_ssl)
        return self._anaconda_client_api

    def token(self):
        """Return the current token registered with authenticate."""
        return self._anaconda_client_api.token

    def load_token(self):
        """Load current authenticated token."""
        token = None
        try:
            if isinstance(self._anaconda_client_api, TeamEditionAPI):
                token = self._anaconda_client_api.load_token()
            else:
                token = binstar_client.utils.load_token(self.get_api_url())
        except OSError:
            pass

        return token

    def _login(  # pylint: disable=too-many-arguments
            self, username, password, application, application_url, verify_ssl=None,
    ):
        """Login callback."""
        if isinstance(self._anaconda_client_api, TeamEditionAPI):
            new_token = self._anaconda_client_api.authenticate(username, password, verify_ssl)
            self._anaconda_client_api.store_token(new_token)

        else:
            new_token = self._anaconda_client_api.authenticate(username, password, application, application_url)

            args = Args()
            args.site = None  # pylint: disable=attribute-defined-outside-init
            args.token = new_token  # pylint: disable=attribute-defined-outside-init

            binstar_client.utils.store_token(new_token, args)

        return new_token

    def login(  # pylint: disable=too-many-arguments
            self, username, password, application, application_url, verify_ssl=None,
    ):
        """Login to anaconda server."""
        method = self._login
        return self._create_worker(method, username, password, application, application_url, verify_ssl)

    def logout(self):
        """
        Logout from anaconda.org.

        This method removes the authentication and removes the token.
        """
        error = None
        args = Args()
        args.site = None  # pylint: disable=attribute-defined-outside-init
        args.token = self.token()  # pylint: disable=attribute-defined-outside-init

        if isinstance(self._anaconda_client_api, TeamEditionAPI):
            self._anaconda_client_api.logout()
        else:
            binstar_client.utils.remove_token(args)

            if self.token():
                try:
                    self._anaconda_client_api.remove_authentication()
                except binstar_client.errors.Unauthorized as e:
                    error = e
                except Exception as e:
                    error = e

            logger.info('logout successful')

        CONF.set_logged_data()
        return error

    def load_repodata(self, repodata, metadata=None, python_version=None):
        """
        Load all the available packages information for downloaded repodata.

        For downloaded repodata files (repo.anaconda.com), additional
        data provided (anaconda cloud), and additional metadata and merge into
        a single set of packages and apps.

        If python_version is not none, exclude all package/versions which
        require an incompatible version of python.

        Parameters
        ----------
        repodata: dict of dicts
            Data loaded from the conda cache directories.
        metadata: dict
            Metadata info form different sources. For now only from
            repo.anaconda.com
        python_version: str
            Python version used in preprocessing.
        """
        method = self._load_repodata
        return self._create_worker(
            method,
            repodata,
            metadata=metadata,
            python_version=python_version,
        )

    def prepare_model_data(self, packages, linked, pip=None, metadata=None):
        """Prepare downloaded package info along with pip pacakges info."""
        method = self._prepare_model_data
        return self._create_worker(
            method,
            packages,
            linked,
            pip=pip,
            metadata=metadata
        )

    def user(self):
        """Return current logged user information."""
        return self.organizations(login=None)

    def domain(self):
        """Return current domain."""
        return self._anaconda_client_api.domain

    def packages(  # pylint: disable=too-many-arguments
            self, login=None, platform=None, package_type=None, type_=None, access=None,
    ):
        """Return all the available packages for a given user.

        Parameters
        ----------
        type_: Optional[str]
            Only find packages that have this conda `type`, (i.e. 'app').
        access : Optional[str]
            Only find packages that have this access level (e.g. 'private',
            'authenticated', 'public').
        """
        method = self._anaconda_client_api.user_packages
        return self._create_worker(
            method,
            login=login,
            platform=platform,
            package_type=package_type,
            type_=type_,
            access=access,
        )

    def organizations(self, login):
        """List all the organizations a user has access to."""
        try:
            user = self._anaconda_client_api.user(login=login)
        except Exception:
            user = {}
        return user

    def get_api_url(self) -> str:
        """Get the anaconda client url configuration."""
        if isinstance(self._anaconda_client_api, TeamEditionAPI):
            return CONF.get('main', 'anaconda_server_api_url')

        config_data: typing.Any = {}
        with anaconda_solvers.catch_and_notify():
            config_data = binstar_client.utils.get_config()

        return config_data.get('url', 'https://api.anaconda.org')

    def get_api_info_url(self) -> str:
        """Get the anaconda client info url configuration."""
        if isinstance(self._anaconda_client_api, TeamEditionAPI):
            return url_utils.join(CONF.get('main', 'anaconda_server_api_url'), 'api/system')

        config_data: typing.Any = {}
        with anaconda_solvers.catch_and_notify():
            config_data = binstar_client.utils.get_config()

        return config_data.get('url', 'https://api.anaconda.org')

    @staticmethod
    def set_api_url(url):
        """Set the anaconda client url configuration."""
        with anaconda_solvers.catch_and_notify():
            config_data = binstar_client.utils.get_config()
            config_data['url'] = url
            binstar_client.utils.set_config(config_data)

    def get_ssl(self, set_conda_ssl: bool = True) -> bool:
        """
        Get conda ssl configuration and set navigator and anaconda-client accordingly.
        """
        config: typing.Any = {}
        with anaconda_solvers.catch_and_notify():
            config = binstar_client.utils.get_config()

        if not set_conda_ssl:
            return config.get('verify_ssl', config.get('ssl_verify', True))

        value = self._conda_api.config_get('ssl_verify').communicate()[0].get('ssl_verify')

        config['verify_ssl'] = config['ssl_verify'] = value
        with anaconda_solvers.catch_and_notify():
            binstar_client.utils.set_config(config)

        logged_api_url: typing.Optional[str] = CONF.get('main', 'logged_api_url', None)
        trusted_servers: typing.List[str] = CONF.get('ssl', 'trusted_servers', [])
        if url_utils.netloc(logged_api_url or '') in trusted_servers:
            # ignore preference update, if user is currently logged into trusted server (overrides ssl_verification
            # preference)
            pass
        elif isinstance(value, bool):
            CONF.set('main', 'ssl_verification', value)
            CONF.set('main', 'ssl_certificate', None)
        else:
            CONF.set('main', 'ssl_verification', True)
            CONF.set('main', 'ssl_certificate', value)

        return value

    def set_ssl(self, value: bool) -> None:
        """Set the anaconda client url configuration."""
        config_data: typing.Any = {}
        with anaconda_solvers.catch_and_notify():
            config_data = binstar_client.utils.get_config()
        config_data['verify_ssl'] = value
        config_data['ssl_verify'] = value
        with anaconda_solvers.catch_and_notify():
            binstar_client.utils.set_config(config_data)

        self._conda_api.config_set('ssl_verify', value).communicate()

    def _get_api_info(self, url, proxy_servers=None, verify=True):
        """Callback."""
        proxy_servers = proxy_servers or {}
        data = {
            'api_url': url,
            'api_docs_url': 'https://api.anaconda.org/docs',
            'brand': AnacondaBrand.DEFAULT,
            'conda_url': 'https://conda.anaconda.org',
            'main_url': 'https://anaconda.org',
            'pypi_url': 'https://pypi.anaconda.org',
            'swagger_url': 'https://api.anaconda.org/swagger.json',
        }
        if is_internet_available():
            try:
                r = requests.get(
                    url,
                    proxies=proxy_servers,
                    verify=api_utils.normalize_certificate(verify),
                    timeout=self.DEFAULT_TIMEOUT,
                )
                http_logger.http(response=r)
                content = to_text_string(r.content, encoding='utf-8')
                new_data = json.loads(content)

                # Enforce no trailing slash
                for key, value in new_data.items():
                    if is_text_string(value):
                        data[key] = value[:-1] if value[-1] == '/' else value

            except Exception as error:
                logger.error(str(error))

        return data

    def get_api_info(self, url, proxy_servers=None, verify=True):
        """Query anaconda api info."""
        proxy_servers = proxy_servers or {}
        method = self._get_api_info
        return self._create_worker(method, url, proxy_servers=proxy_servers, verify=verify)


CLIENT_API = None


def ClientAPI():
    """Client API threaded worker."""
    global CLIENT_API  # pylint: disable=global-statement

    if CLIENT_API is None:
        CLIENT_API = _ClientAPI()

    return CLIENT_API
