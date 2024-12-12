# -*- coding: utf8 -*-

"""Utility components for telemetry."""

from __future__ import annotations

__all__ = ['ApiClient', 'Stats']

import abc
import collections.abc
import functools
import locale
import platform
import sys
import typing
import urllib.parse

from qtpy import QtCore
from qtpy import QtWidgets
import requests.cookies

from anaconda_navigator import __about__

if typing.TYPE_CHECKING:
    from anaconda_navigator.api import anaconda_api


T = typing.TypeVar('T')


def normalize_root_url(value: str) -> str:
    """
    Make sure that path is normalized to be used as a root URL.

    It strips down all query params, fragment and ensures that URL ends with a :code:`/`. The latter is required for the
    :func:`~urllib.parse.urljoin`.
    """
    scheme: str
    netloc: str
    path: str
    scheme, netloc, path, _, _ = urllib.parse.urlsplit(value)

    if not path.endswith('/'):
        path += '/'

    if not netloc:
        if path[:1] in '/':
            raise ValueError(f'{value!r} is not a valid root url')
        netloc, path = path.split('/', 1)

    return urllib.parse.urlunsplit((scheme or 'https', netloc, path, '', ''))


class ApiClient(metaclass=abc.ABCMeta):  # pylint: disable=too-few-public-methods
    """
    Base for API client classes.

    :param root_url: Base URL for the API.
    """

    __slots__ = ('__session', '__urls')

    def __init__(self, root_url: str) -> None:
        """Initialize new instance of a :class:`ApiClient`."""
        self.__session: typing.Final[requests.Session] = requests.Session()
        self.__urls: typing.Final[typing.Dict[str, str]] = {'': normalize_root_url(root_url)}

    @property
    def _cookies(self) -> requests.cookies.RequestsCookieJar:  # noqa: D401
        """Cookies attached to each request."""
        return self.__session.cookies

    @property
    def _headers(self) -> typing.MutableMapping[str, typing.Union[bytes, str]]:  # noqa: D401
        """Headers attached to each request."""
        return self.__session.headers

    def _request(
            self,
            method: str,
            url: str,
            *args: typing.Any,
            ignore_status: bool = False,
            **kwargs: typing.Any,
    ) -> requests.Response:
        """
        Send request to the service.

        :param method: HTTP method for the request.
        :param url: Path to the API endpoint. It should be a URL relative to the :code:`root_url`.
        :param args: Extra positional arguments to :func:`requests.request`.
        :param ignore_status: Do not raise an error if server responds with non-200 status code.
        :param kwargs: Extra keyword arguments to :func:`requests.request`.
        :return: Server response.
        """
        target: typing.Optional[str] = self.__urls.get(url, None)
        if target is None:
            target = self.__urls[url] = urllib.parse.urljoin(self.__urls[''], url)

        response: requests.Response = self.__session.request(method, target, *args, **kwargs)
        if not ignore_status:
            response.raise_for_status()
        return response


MACHINE_CLARIFICATION: typing.Final[collections.abc.Mapping[str, str]] = {
    'amd64': '64',
    'x86': '32',
    'x86_64': '64',
}
PLATFORM_CLARIFICATION: typing.Final[tuple[tuple[str, str], ...]] = (
    ('darwin', 'osx'),
    ('win32', 'win'),
)


StatsDetails = typing.TypedDict('StatsDetails', {
    'conda-version': str,
    'locale-encoding': str,
    'locale-language': str,
    'navigator-version': str,
    'os-long': str,
    'os-short': str,
    'platform': str,
    'python-long': str,
    'python-short': str,
    'qt-version': str,
    'screen-height': int,
    'screen-width': int,
})


class Stats:
    """Helper class to collect basic details about the execution environment."""

    @functools.cached_property
    def conda_version(self) -> str:  # noqa: D401
        """Conda version that Navigator works with."""
        return self._api.conda_package_version(pkg='conda', name='root') or 'unknown'

    @functools.cached_property
    def details(self) -> StatsDetails:  # noqa: D401
        """
        Collected details in a dictionary format.

        Can be used to send statistics to the server.
        """
        return {
            'conda-version': self.conda_version,
            'locale-encoding': self.locale_encoding,
            'locale-language': self.locale_language,
            'navigator-version': self.navigator_version,
            'os-long': self.os_long,
            'os-short': self.os_short,
            'platform': self.platform,
            'python-long': self.python_long,
            'python-short': self.python_short,
            'qt-version': self.qt_version,
            'screen-height': self.screen_height,
            'screen-width': self.screen_width,
        }

    @functools.cached_property
    def locale_encoding(self) -> str:  # noqa: D401
        """Encoding used for the current locale."""
        return self._locale[1] or 'unknown'

    @functools.cached_property
    def locale_language(self) -> str:  # noqa: D401
        """Language of the current locale."""
        return self._locale[0] or 'unknown'

    @functools.cached_property
    def navigator_version(self) -> str:  # noqa: D401
        """Version of the Navigator."""
        return __about__.__version__

    @functools.cached_property
    def os_long(self) -> str:  # noqa: D401
        """Name of the OS in a long form (including version)."""
        return f'{self.os_short} {platform.release()}'

    @functools.cached_property
    def os_short(self) -> str:  # noqa: D401
        """Name of the OS in a short form (without version)."""
        return platform.system()

    @functools.cached_property
    def platform(self) -> str:  # noqa: D401
        """Platform that Navigator runs in."""
        architecture: str = platform.machine().lower()
        architecture = MACHINE_CLARIFICATION.get(architecture, architecture)

        left: str
        right: str
        system: str = sys.platform.lower()
        for left, right in PLATFORM_CLARIFICATION:
            if system.startswith(left):
                system = right
                break

        return f'{system}-{architecture}'

    @functools.cached_property
    def pyqt_version(self) -> str:  # noqa: D401
        """Version of the PyQt used to display Navigator."""
        return (
                self._api.conda_package_version(pkg='pyqt5', name='root') or
                self._api.conda_package_version(pkg='pyqt4', name='root') or
                self._api.conda_package_version(pkg='pyqt', name='root') or
                'unknown'
        )

    @functools.cached_property
    def python_long(self) -> str:  # noqa: D401
        """Current version of the python with extra details."""
        return f'{platform.python_version()} ({platform.python_implementation()})'

    @functools.cached_property
    def python_short(self) -> str:  # noqa: D401
        """Short version of the python (without patch and extra details)."""
        return '.'.join(platform.python_version_tuple()[:2])

    @functools.cached_property
    def qt_version(self) -> str:  # noqa: D401
        """Version of the Qt that Navigator is running in."""
        try:
            return QtCore.__version__
        except AttributeError:
            return 'unknown'

    @functools.cached_property
    def screen_height(self) -> int:  # noqa: D401
        """Height of the primary screen."""
        return self._screen[1]

    @functools.cached_property
    def screen_width(self) -> int:  # noqa: D401
        """Width of the primary screen."""
        return self._screen[0]

    @functools.cached_property
    def _api(self) -> anaconda_api._AnacondaAPI:  # noqa: D401
        """Shortcut to :code:`AnacondaAPI`."""
        from anaconda_navigator.api import anaconda_api  # pylint: disable=import-outside-toplevel
        return anaconda_api.AnacondaAPI()

    @functools.cached_property
    def _locale(self) -> tuple[str | None, str | None]:  # noqa: D401
        """Collected details on current locale."""
        return locale.getlocale()

    @functools.cached_property
    def _screen(self) -> tuple[int, int]:  # noqa: D401
        """Collected details on screen dimensions."""
        try:
            widget: QtWidgets.QDesktopWidget = QtWidgets.QDesktopWidget()
            geometry: QtCore.QRect = widget.screenGeometry(widget.primaryScreen())
            return geometry.width(), geometry.height()
        except Exception:  # pylint: disable=broad-exception-caught
            return 0, 0
