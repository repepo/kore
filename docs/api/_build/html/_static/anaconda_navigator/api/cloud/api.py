# -*- coding: utf-8 -*-

"""Client for Cloud API"""

from __future__ import annotations

__all__ = ['EnvironmentSort', '_CloudAPI', 'CloudAPI']

import collections.abc
import enum
import os
import shutil
import tempfile
import typing
from urllib import parse

import anaconda_cloud_auth.handlers
from qtpy import QtCore
import requests

from anaconda_navigator import config as navigator_config
from anaconda_navigator.utils import workers
from . import utilities

if typing.TYPE_CHECKING:
    from anaconda_navigator.config import user as user_config
    from . import types


def cancel_login():
    """
    Cancel login process.

    This should kill server started by :meth:`~_CloudAPI.login`.
    """
    try:
        anaconda_cloud_auth.handlers.shutdown_all_servers()
    except Exception:  # nosec B110 # pylint: disable=broad-exception-caught
        pass


class EnvironmentSort(str, enum.Enum):
    """Options for environment sorting."""

    NAME_ASC = 'name'
    NAME_DESC = '-name'

    CREATED_ASC = 'created_at'
    CREATED_DESC = '-created_at'

    UPDATED_ASC = 'updated_at'
    UPDATED_DESC = '-updated_at'


class _CloudAPI(QtCore.QObject):
    """
    Anaconda Cloud API.

    :param: Navigator's config object.
    """

    sig_token_changed = QtCore.Signal()

    def __init__(self, config: typing.Optional[user_config.UserConfig] = None) -> None:
        """Initialize new :class:`~CloudAPI` instance."""
        super().__init__()

        if config is None:
            config = navigator_config.CONF

        self.__config: typing.Final[user_config.UserConfig] = config
        self.__token: typing.Optional[str] = None
        self.__account: collections.abc.Mapping[str, typing.Any] = {}

        self.__routes: typing.Final[utilities.EndpointCollection] = utilities.EndpointCollection(
            base_url=self.__config.get('main', 'cloud_base_url') or '',
            endpoints={
                'authentication': {
                    'login': 'api/iam/token',
                    'logout': 'api/iam/logout',
                },
                'basics': {
                    'ping': '',
                    'account': 'api/account'
                },
                'environments': {
                    'list_environments': 'api/environments/my',
                    'create_environment': 'api/environments/my',
                    'update_environment': 'api/environments/my/{name}',
                    'delete_environment': 'api/environments/my/{name}',
                    'download_environment': 'api/environments/my/{name}.yml',
                },
            },
        )

        self.__session: typing.Final[utilities.Session] = utilities.Session()

        self.sig_token_changed.connect(self.__refresh_session)
        self.__load_token()

    def __set_token(self, token: str | None) -> None:
        """Update token value."""
        self.__token = token
        self.sig_token_changed.emit()

    def __load_token(self) -> bool:
        """Try to load exising token in the keyring."""
        access_token: str | None = None
        try:
            token_info: typing.Final[anaconda_cloud_auth.client.TokenInfo] = anaconda_cloud_auth.client.TokenInfo.load(
                domain=self.__config.get('main', 'auth_domain', anaconda_cloud_auth.client.AuthConfig().domain),
            )
            access_token = token_info.get_access_token()
        except Exception:  # nosec B110 # pylint: disable=broad-exception-caught
            pass

        self.__set_token(access_token)
        return bool(access_token)

    def __refresh_session(self) -> None:
        """Inject authorization details into session."""
        self.__session.headers['Authorization'] = f'Bearer {self.token or ""}'
        self.__account = self.account()

    @property
    def token(self) -> typing.Optional[str]:  # noqa: D401
        """Current authorization token."""
        return self.__token

    @property
    def username(self) -> str:  # noqa: D401
        """Current authorization token."""
        return str((self.__account or {}).get('user', {}).get('email', ''))

    # ╠═════════════════════════════════════════════════════════════════════════════════════════════╡ Authentication ╞═╣

    @workers.Task(workers.AddCancelCallback(cancel_login))
    def login(self) -> None:
        """Perform user authorization."""
        try:
            anaconda_cloud_auth.login()
        except Exception as error:
            raise workers.TaskFailedError() from error
        if not self.__load_token():
            raise workers.TaskFailedError()

    @workers.Task
    def ping(self) -> bool:
        """Check if API server is available."""
        try:
            return self.__session.request(
                'GET',
                self.__routes.basics.ping,
                allow_redirects=False,
                timeout=5,
            ).status_code < 500
        except requests.ConnectionError:
            return False

    @workers.Task
    def account(self) -> collections.abc.Mapping[str, typing.Any]:
        """Check if API server is available."""
        return self.__session.request(
            'GET',
            self.__routes.basics.account,
            allow_redirects=False,
            timeout=5,
        ).json()

    @workers.Task
    def logout(self) -> None:
        """Remove token data and clear session."""
        anaconda_cloud_auth.logout()
        self.__set_token(None)

    # ╠═══════════════════════════════════════════════════════════════════════════════════════════════╡ Environments ╞═╣

    @workers.Task
    def list_environments(
            self,
            limit: int = 100,
            offset: int = 0,
            sort: EnvironmentSort = EnvironmentSort.NAME_ASC,
    ) -> 'types.CloudEnvironmentCollection':
        """
        List available environments for current user.

        :param limit: Maximum number of environments to fetch in a single call.
        :param offset: Number of environments to skip from the start.
        :param sort: How environments in the result should be sorted.
        :return: Collection of environments available to user.
        """
        return self.__session.request(
            'GET',
            self.__routes.environments.list_environments,
            params={
                'limit': limit,
                'offset': offset,
                'sort': sort,
            },
            raise_for_status=True,
        ).json()

    @workers.Task
    def create_environment(self, name: str, path: str) -> None:
        """
        Create a new environment in Cloud.

        :param name: Name of the environment to create.
        :param path: Path to the exported environment (yaml file).
        """
        file_descriptor: int
        file_path: str
        file_descriptor, file_path = tempfile.mkstemp()

        file_stream: typing.TextIO
        with os.fdopen(file_descriptor, 'wt+', encoding='utf-8') as file_stream:
            file_writer: utilities.JsonWriter = utilities.JsonWriter(file_stream)

            stream: typing.TextIO
            with open(path, 'rt', encoding='utf-8') as stream:
                file_writer.serialize({'name': name, 'yaml': stream})

            file_stream.seek(0, 0)
            self.__session.request(
                'POST',
                self.__routes.environments.create_environment,
                data=file_stream,
                raise_for_status=True,
            )

        os.remove(file_path)

    @workers.Task
    def update_environment(self, name: str, path: str, rename_to: typing.Optional[str] = None) -> None:
        """
        Update environment in Cloud.

        :param name: Name of the environment to update.
        :param path: Path to the exported environment (yaml file).
        :param rename_to: Optional name to change Cloud environment to.
        """
        if rename_to is None:
            rename_to = name

        file_descriptor: int
        file_path: str
        file_descriptor, file_path = tempfile.mkstemp()

        file_stream: typing.TextIO
        with os.fdopen(file_descriptor, 'wt+', encoding='utf-8') as file_stream:
            file_writer: utilities.JsonWriter = utilities.JsonWriter(file_stream)

            stream: typing.TextIO
            with open(path, 'rt', encoding='utf-8') as stream:
                file_writer.serialize({'name': rename_to, 'yaml': stream})

            file_stream.seek(0, 0)

            self.__session.request(
                'PUT',
                self.__routes.environments.update_environment.format(
                    name=parse.quote(string=name, safe=''),
                ),
                data=file_stream,
                raise_for_status=True,
            )

        os.remove(file_path)

    @workers.Task
    def delete_environment(self, name: str) -> None:
        """
        Remove environment from Cloud.

        :param name: Name of the environment to remove.
        """
        self.__session.request(
            'DELETE',
            self.__routes.environments.delete_environment.format(
                name=parse.quote(string=name, safe=''),
            ),
            raise_for_status=True,
        )

    @workers.Task
    def download_environment(self, name: str, path: str) -> None:
        """
        Download environment description file from the Cloud.

        :param name: Name of the environment to download.
        :param path: Path to store downloaded environment to.
        """
        response: 'requests.Response' = self.__session.request(
            'GET',
            self.__routes.environments.download_environment.format(
                name=parse.quote(string=name, safe=''),
            ),
            raise_for_status=True,
            stream=True,
        )

        stream: typing.BinaryIO
        with open(path, 'wb') as stream:
            shutil.copyfileobj(response.raw, stream)


CLOUD_API_INSTANCE: typing.Optional[_CloudAPI] = None


def CloudAPI() -> _CloudAPI:  # pylint: disable=invalid-name
    """Retrieve :class:`~_CloudAPI` instance."""
    global CLOUD_API_INSTANCE  # pylint: disable=global-statement
    if CLOUD_API_INSTANCE is None:
        CLOUD_API_INSTANCE = _CloudAPI()
    return CLOUD_API_INSTANCE
