# -*- coding: utf-8 -*-

"""Simplistic thread-safe HTTP sessions."""

__all__ = ['SessionOptions', 'Session']

import platform
import threading
import typing

import requests

from anaconda_navigator import __about__
from anaconda_navigator.utils.logs.loggers import http_logger


KeyT = typing.TypeVar('KeyT')
ValueT = typing.TypeVar('ValueT')


def merge_dicts(*args: typing.Mapping[KeyT, ValueT]) -> typing.Dict[KeyT, ValueT]:
    """Combine values of multiple mappings into a single dictionary."""
    arg: typing.Mapping[KeyT, ValueT]
    result: typing.Dict[KeyT, ValueT] = {}
    for arg in args:
        result.update(arg)
    return result


class SessionOptions(typing.Generic[KeyT, ValueT]):
    """
    Proxy-mapping, which allows editing common session options in a thread-safe way.

    Each access to this collection must be independent and atomic, as it might be changed between the calls in other
    thread.

    :param lock: Parent lock to use for accessing wrapped content.
    :param content: Content to control access to.
    """

    __slots__ = ('__lock', '__content')

    def __init__(self, lock: threading.Lock, content: typing.MutableMapping[KeyT, ValueT]) -> None:
        """Initialize new :class:`~SessionOptions` instance."""
        self.__lock: typing.Final[threading.Lock] = lock
        self.__content: typing.Final[typing.MutableMapping[KeyT, ValueT]] = content

    def clear(self) -> None:
        """Clear content."""
        with self.__lock:
            self.__content.clear()

    def copy(self) -> typing.Dict[KeyT, ValueT]:
        """Create a copy of content."""
        with self.__lock:
            return dict(self.__content)

    def update(
            self,
            content: typing.Union[
                None,
                typing.Mapping[KeyT, ValueT],
                typing.Iterable[typing.Tuple[KeyT, ValueT]],
            ] = None,
            **kwargs: ValueT
    ) -> None:
        """Add new records to the content."""
        with self.__lock:
            if content is None:
                self.__content.update(**kwargs)
            else:
                self.__content.update(content, **kwargs)

    def __getitem__(self, key: KeyT) -> ValueT:
        """Execute wrapped `__getitem__` method in a thread-safe way."""
        with self.__lock:
            return self.__content[key]

    def __setitem__(self, key: KeyT, value: ValueT) -> None:
        """Execute wrapped `__setitem__` method in a thread-safe way."""
        with self.__lock:
            self.__content[key] = value

    def __delitem__(self, key: KeyT) -> None:
        """Execute wrapped `__delitem__` method in a thread-safe way."""
        with self.__lock:
            del self.__content[key]


class Session:
    """
    Requests wrapper with common configuration for requests.

    This wrapper is created to ensure thread-safe operation, as :class:`~requests.Session` might have some issues in
    this area.
    """

    __slots__ = ('__lock', '__cookies', '__headers')

    def __init__(self) -> None:
        """Initialize new :class:`~Session` instance."""
        self.__lock: typing.Final[threading.Lock] = threading.Lock()
        self.__cookies: typing.Final[typing.Dict[str, str]] = {}
        self.__headers: typing.Final[typing.Dict[str, str]] = {
            'Authorization': '',
            'User-Agent': ' '.join([
                'Mozilla/5.0',
                f'({platform.system()} {platform.machine()})',
                f'Python/{platform.python_version()}',
                f'requests/{requests.__version__}',
                f'Navigator/{__about__.__version__}',
            ]),
        }

    @property
    def cookies(self) -> 'SessionOptions[str, str]':  # noqa: D401
        """Collection of cookies used by all requests."""
        return SessionOptions(lock=self.__lock, content=self.__cookies)

    @property
    def headers(self) -> 'SessionOptions[str, str]':  # noqa: D401
        """Collection of headers used by all requests."""
        return SessionOptions(lock=self.__lock, content=self.__headers)

    def request(
            self,
            method: typing.Union[str, bytes],
            url: typing.Union[str, bytes],
            *,
            cookies: typing.Optional[typing.Mapping[str, str]] = None,
            headers: typing.Optional[typing.Mapping[str, str]] = None,
            raise_for_status: bool = False,
            **kwargs: typing.Any,
    ) -> requests.Response:
        """
        Send a new HTTP(s) request.

        :param method: HTTP method to use for request (:code:`'GET'`, :code:`'POST'`, etc.).
        :param url: URL to send request to.
        :param cookies: Collection of additional cookies to use with this request.
        :param headers: Collection of additional headers to use with this request.
        :param raise_for_status: Raise exception for non-200 results.
        :param kwargs: Additional options to use with :mod:`requests`.
        :return: HTTP response details.
        """
        kwargs.setdefault('timeout', 30)

        if cookies is None:
            cookies = {}
        if headers is None:
            headers = {}

        with self.__lock:
            cookies = merge_dicts(self.__cookies, cookies)
            headers = merge_dicts(self.__headers, headers)

        result: requests.Response = requests.request(  # pylint: disable=missing-timeout
            method,
            url,
            cookies=cookies,
            headers=headers,
            **kwargs,
        )
        http_logger.http(response=result)

        if raise_for_status:
            result.raise_for_status()
        return result
