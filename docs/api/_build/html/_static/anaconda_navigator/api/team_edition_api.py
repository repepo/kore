# -*- coding: utf-8 -*-

# pylint: disable=invalid-name,missing-module-docstring

import collections
import datetime
import json
import os
import stat
import traceback
import typing
import urllib.parse
import uuid
from enum import Enum
from itertools import product
from os.path import exists, isdir, isfile, join
from urllib.parse import quote_plus

import repo_cli
import requests
from binstar_client.utils.config import TOKEN_DIRS

from anaconda_navigator.utils import url_utils
from anaconda_navigator.utils.logs.loggers import http_logger as logger
from anaconda_navigator.config import CONF
from . import utils as api_utils

if typing.TYPE_CHECKING:
    from anaconda_navigator.config import user as user_config

Args = collections.namedtuple('Args', ['site'])
SSL_ERROR_MESSAGE = (
    'Your SSL certificate is self-signed or broken and cannot be validated. '  # pylint: disable=implicit-str-concat
    'Please change the certificate or set “ssl_verify:false” in your .condarc configuration file.'
)


class TokenFileExtension(Enum):  # pylint: disable=missing-class-docstring
    BEARER: str = 'token'
    JWT: str = 'jwt'


def inject_url_query(url: str, arguments: typing.Mapping[str, typing.Any]) -> str:
    """Replace query part in the `url`."""
    scheme: str
    netloc: str
    path: str
    query: str
    fragment: str
    scheme, netloc, path, query, fragment = urllib.parse.urlsplit(url)
    query = urllib.parse.urlencode({**urllib.parse.parse_qs(query, keep_blank_values=True), **arguments}, doseq=True)
    return urllib.parse.urlunsplit((scheme, netloc, path, query, fragment))


class TeamEditionAPI:
    """
    This is the interface for working with Anaconda Server API.
    """
    DEFAULT_CONTENT_TYPE = {'Content-Type': 'application/json'}

    def __init__(
            self,
            base_url: str,
            verify_ssl: typing.Union[None, bool, str] = False,
    ) -> None:
        if verify_ssl is None:
            verify_ssl = url_utils.netloc(base_url) in CONF.get('ssl', 'trusted_servers', [])

        self._urls = {
            'api': url_utils.join(base_url, 'api'),
            'repo': url_utils.join(base_url, 'api/repo'),
            'login': url_utils.join(base_url, 'api/auth/login'),
            'account': url_utils.join(base_url, 'api/account'),
            'tokens': url_utils.join(base_url, 'api/account/tokens'),
            'channels': url_utils.join(base_url, 'api/account/channels'),
            'all_channels': url_utils.join(base_url, 'api/channels'),
            'system': url_utils.join(base_url, 'api/system')
        }
        self._verify_ssl = verify_ssl
        self.token = None
        self.domain = base_url

    def ping(self) -> bool:
        """Check for Anaconda Server server availability."""
        try:
            response: requests.Response = requests.get(
                self._urls['system'],
                verify=api_utils.normalize_certificate(self._verify_ssl),
                timeout=5,
            )
            logger.http(response=response)
            response.raise_for_status()
            healthy = True
        except (requests.HTTPError, requests.ConnectionError):
            healthy = False

        return healthy

    def authenticate(
            self,
            username: str,
            password: str,
            verify_ssl: typing.Union[None, bool, str] = None,
    ) -> str:
        """
        Login using direct grant and returns the JWT token.

        :param username: The username to authenticate.
        :param password: The password to authenticate.
        :param verify_ssl: The path to ssl certificate or flag to turn off validation.

        :return: The dumped json string.
        """
        if verify_ssl is None:
            verify_ssl = self._verify_ssl

        data = {
            'username': username,
            'password': password,
        }
        error_text = None

        token: typing.Optional[str] = None
        refresh_token: typing.Optional[str] = None
        try:
            resp = requests.post(
                self._urls['login'],
                data=json.dumps(data),
                headers=self.DEFAULT_CONTENT_TYPE,
                verify=api_utils.normalize_certificate(verify_ssl),
                timeout=60,
            )
            logger.http(response=resp)
            resp.raise_for_status()
            response_data = resp.json()
            token = response_data['token']
            refresh_token = response_data['refresh_token']

        except requests.exceptions.SSLError:
            error_text = SSL_ERROR_MESSAGE

        except requests.HTTPError as error:
            error_text = 'Invalid Credentials!' if error.response.status_code == 401 else str(error)

        except Exception:  # pylint: disable=broad-except
            logger.error(
                'Exception happened during the login into Anaconda Server. Traceback: %s', traceback.format_exc())

            base_filename: str = logger.handlers[0].baseFilename  # type: ignore
            error_text = (
                'An unexpected error happened! '
                f'Please see logs at {base_filename} and contact your system administrator.'
            )

        if error_text:
            raise Exception(error_text)   # pylint: disable=broad-exception-raised

        return json.dumps({
            'token': token,
            'refresh_token': refresh_token,
            'jwt_token': token,
            'jwt_token_refresh': refresh_token
        })

    def logout(self):  # pylint: disable=missing-function-docstring
        self.remove_token()

    @staticmethod
    def _get_team_edition_api_url():
        return f'jwt_{CONF.get("main", "anaconda_server_api_url")}'

    @staticmethod
    def _get_token_file_path(directory, name, extension=TokenFileExtension.BEARER):
        return join(directory, f'{quote_plus(name)}.{extension.value}')

    def load_token(self):
        """
        Loads the JWT token to be used in further requests to authenticate user.

        :return str:
        """
        for token_dir in TOKEN_DIRS:
            url = self._get_team_edition_api_url()

            tokenfile = self._get_token_file_path(token_dir, url, TokenFileExtension.JWT)
            _is_file = exists(tokenfile)

            if _is_file:
                with open(tokenfile) as fd:  # pylint: disable=unspecified-encoding
                    token = fd.read().strip()
                if token:
                    return token
                os.unlink(tokenfile)

        return '{}'

    def store_binstar_token(self, token):
        """
        Stores the binstar token required to work with Anaconda Server
        through 'native' conda interface.

        :param str token: The string as a token.
        """
        files = []
        for token_dir in TOKEN_DIRS:
            if not isdir(token_dir):
                os.makedirs(token_dir)

            files.extend((
                # repo-cli compatibility case.
                # API endpoint is used as token file name, but instead of quoted URL CLI uses os.path.join()
                self._get_token_file_path(token_dir, join(self._urls['api'], 'repo')),
                self._get_token_file_path(token_dir, self._urls['api'])
            ))

        for tokenfile in files:
            if isfile(tokenfile):
                os.unlink(tokenfile)

            with open(tokenfile, 'w') as fd:  # pylint: disable=unspecified-encoding
                fd.write(token)
            os.chmod(tokenfile, stat.S_IWRITE | stat.S_IREAD)

    def store_token(self, token):
        """
        Stores the passed JWT token locally to be used in further
        user authentication.

        :param str token: The json data dumped in the string.
        """
        for token_dir in TOKEN_DIRS:
            url = self._get_team_edition_api_url()

            if not isdir(token_dir):
                os.makedirs(token_dir)
            tokenfile = self._get_token_file_path(token_dir, url, TokenFileExtension.JWT)

            if isfile(tokenfile):
                os.unlink(tokenfile)

            with open(tokenfile, 'w') as fd:  # pylint: disable=unspecified-encoding
                fd.write(token)
            os.chmod(tokenfile, stat.S_IWRITE | stat.S_IREAD)

    def remove_token(self):
        """
        Removes the existing JWT token from local space.
        """
        jwt_url = self._get_team_edition_api_url()
        url = self._get_team_edition_api_url().replace('jwt_', '')

        url_suffixes = [
            url,
            jwt_url,
            self._urls['api'],
            join(self._urls['api'], 'repo')
        ]
        for token_dir, url_suffix, extension in product(TOKEN_DIRS, url_suffixes, TokenFileExtension):
            tokenfile = self._get_token_file_path(token_dir, url_suffix, extension)
            if isfile(tokenfile):
                os.unlink(tokenfile)

        try:
            args = Args(None)
            repo_cli.utils.config.remove_token(args)
        except TypeError:
            pass

    def __http_request(
            self,
            url: str,
            method: str = 'GET',
            *,
            cookies: typing.Optional[typing.Mapping[str, typing.Any]] = None,
            headers: typing.Optional[typing.Mapping[str, typing.Any]] = None,
            verify: typing.Union[None, bool, str] = None,
            raise_for_error: bool = True,
    ) -> requests.Response:
        """
        Process HTTP request.

        :param url: URL to send request to.
        :param method: Method to send URL with. By default: "GET".
        :param cookies: Additional cookies, if necessary.
        :param headers: Additional headers, if necessary.
        :param verify: Custom verification options.
        :param raise_for_error: Raise error if request fails.
        :return: HTTP response.
        """
        r_headers: typing.Dict[str, typing.Any] = {**self.DEFAULT_CONTENT_TYPE}
        r_cookies: typing.Dict[str, typing.Any] = {}
        if verify is None:
            verify = self._verify_ssl

        token: typing.Any = json.loads(self.load_token())
        if token:
            r_cookies.update(token)
        else:
            r_headers['X-Auth'] = self.token or CONF.get('main', 'anaconda_server_token')

        if headers is not None:
            r_headers.update(headers)
        if cookies is not None:
            r_cookies.update(cookies)

        result: requests.Response = requests.request(
            method,
            url,
            headers=r_headers,
            cookies=r_cookies,
            verify=verify,
            timeout=60,
        )
        if raise_for_error:
            result.raise_for_status()
        logger.http(response=result)
        return result

    def _get_user_data(self):
        return self.__http_request(self._urls['account']).json()

    def user(self, login: typing.Optional[str] = None) -> typing.Any:  # pylint: disable=unused-argument
        """
        Gets the user account info.

        :return dict[str, str]: The user data.
        """
        data: typing.Any = self._get_user_data()
        data['login'] = data.pop('username')

        return data

    def get_user_id(self) -> str:
        """
        Gets the user id.

        :return str: The user id.
        """
        data: typing.Any = self._get_user_data()
        return data.get('user_id', '')

    def create_access_token(self, token):
        """
        Creates the access token (binstar) for the Anaconda Server API to provide
        access to channels listed in the repo.

        Returns the token and the token id.

        :param dict[str, str] token: The JWT token to be used for authentication.

        :return dict: The access token info.
        """
        request_data = {
            'name': f'navigator-token-{datetime.datetime.now().strftime("%Y-%m-%d")}-{uuid.uuid4()}',
            'expires_at': (datetime.datetime.now() + datetime.timedelta(days=365)).strftime('%Y-%m-%d'),
            'scopes': [
                'channel:view', 'channel:view-artifacts', 'subchannel:view', 'subchannel:view-artifacts',
                'artifact:view', 'artifact:download'
            ]
        }

        resp = requests.post(
            self._urls['tokens'],
            headers=self.DEFAULT_CONTENT_TYPE,
            data=json.dumps(request_data),
            cookies=token,
            verify=api_utils.normalize_certificate(self._verify_ssl),
            timeout=60,
        )
        logger.http(response=resp)
        response_data = resp.json()
        self.store_binstar_token(response_data.get('token'))

        return response_data

    def remove_access_token(self, access_token_id):
        """
        Deletes the created access token (binstar) from the Anaconda Server database.

        :param str access_token_id: The token ID to be removed.
        """
        token = json.loads(self.load_token())

        resp = requests.delete(
            f'{self._urls["tokens"]}/{access_token_id}',
            headers=self.DEFAULT_CONTENT_TYPE,
            cookies=token,
            verify=api_utils.normalize_certificate(self._verify_ssl),
            timeout=60,
        )
        logger.http(response=resp)

    def get_channels(self) -> typing.List[typing.Dict[str, typing.Any]]:
        """
        Returns all the channels which are accessible for the user.

        :return: List of dictionaries with available channels data.
        """
        result: typing.List[typing.Dict[str, typing.Any]] = []

        channels_url: str
        if CONF.get('main', 'anaconda_server_show_hidden_channels'):
            channels_url = self._urls['all_channels']
        else:
            channels_url = self._urls['channels']

        offset: int = 0
        step: int = 1000
        total: typing.Optional[int] = None
        while (total is None) or (offset < total):
            url: str = inject_url_query(
                url=channels_url,
                arguments={'offset': offset, 'limit': step, 'sort': 'name,updated_at', 'include_subchannels': 'true'},
            )
            content: typing.Mapping[str, typing.Any] = self.__http_request(url).json()
            if total is None:
                total = content['total_count']
            result.extend(content['items'])

            offset += step

        return result
