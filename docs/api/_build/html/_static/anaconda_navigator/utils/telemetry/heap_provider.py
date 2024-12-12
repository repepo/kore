# -*- coding: utf8 -*-

"""Integration with Heap analytics."""

from __future__ import annotations

__all__ = ['HeapProvider']

import itertools

import collections.abc
import typing

from . import core
from . import utilities

if typing.TYPE_CHECKING:
    import typing_extensions


T = typing.TypeVar('T')

Item: typing_extensions.TypeAlias = 'collections.abc.Mapping[str, typing.Any]'
Source: typing_extensions.TypeAlias = 'T | collections.abc.Iterable[T]'


class AccountProperties(typing.TypedDict, total=False):
    """Body for the :code:`add_account_properties` request."""

    account_id: typing_extensions.Required[str]
    properties: dict[str, str | int]


class Event(typing.TypedDict, total=False):
    """Body for the :code:`track` request."""

    identity: typing_extensions.Required[str]
    event: typing_extensions.Required[str]
    properties: dict[str, str | int]
    timestamp: str
    idempotency_key: str


class UserProperties(typing.TypedDict, total=False):
    """Body for the :code:`add_user_properties` request."""

    identity: typing_extensions.Required[str]
    properties: dict[str, str | int]


class ApiClient(utilities.ApiClient):
    """
    Client to the Heap analytics API.

    :param app_id: ID of the Heap Application to use for tracking.
    """

    __slots__ = ('__credentials',)

    def __init__(self, app_id: str, *, root_url: str = 'https://heapanalytics.com/api/') -> None:
        """Initialize new instance of a :class:`~ApiClient`."""
        super().__init__(root_url)

        self.__credentials: typing.Final[collections.abc.Mapping[str, str]] = {
            'app_id': str(app_id),
        }

    def add_account_properties(
            self,
            accounts: Source[AccountProperties],  # pylint: disable=invalid-sequence-index
    ) -> bool:
        """
        Attach custom account properties to users.

        More details: `Add Account Properties <https://developers.heap.io/reference/add-account-properties>`_.
        """
        return self.__collect('add_account_properties', 'accounts', accounts)

    def add_user_properties(self, users: Source[UserProperties]) -> bool:  # pylint: disable=invalid-sequence-index
        """
        Attach custom properties to any identified users from your servers.

        More details: `Add User Properties <https://developers.heap.io/reference/add-user-properties>`_.
        """
        return self.__collect('add_user_properties', 'users', users)

    def track(self, events: Source[Event]) -> bool:  # pylint: disable=invalid-sequence-index
        """
        Send custom events to Heap server-side.

        More details: `Add User Properties <https://developers.heap.io/reference/track-1>`_.
        """
        return self.__collect('track', 'events', events)

    def __collect(self, url: str, name: str, content: Source[Item]) -> bool:  # pylint: disable=invalid-sequence-index
        """Send request of a common format to the Heap."""
        query: dict[str, typing.Any] = {**self.__credentials}
        if isinstance(content, collections.abc.Mapping):
            query.update(content)
        else:
            query[name] = list(content)

        try:
            self._request('POST', url, json=query)
        except IOError:
            return False
        return True


class HeapProvider(core.Provider):
    """
    Custom provider to integrate with Heap Analytics.

    :param app_id: ID of the Heap Application to use for tracking.
    :param identity: Identity of the current user to track.
    :param alias: Custom alias of the provider.
    """

    __slots__ = ('__api',)

    def __init__(
            self,
            context: core.Context,
            app_id: str,
            *,
            alias: str = 'heap',
            root_url: str = 'https://heapanalytics.com/api/',
    ) -> None:
        """Initialize new instance of a :class:`~HeapProvider`."""
        super().__init__(alias=alias, context=context)

        self.__api: typing.Final[ApiClient] = ApiClient(app_id=app_id, root_url=root_url)

    def close(self) -> None:
        """Close provider."""

    def process(self, events: collections.abc.MutableSequence[core.Event]) -> None:
        """Process all pending events."""
        event: core.Event
        query_user_properties: dict[str, str | int]
        query_events: list[Event] | Event
        while events:
            query_user_properties = {}
            query_events = []
            for event in itertools.islice(events, 0, 1000):
                query_events.append({
                    'event': event.name,
                    'idempotency_key': event.key,
                    'identity': self.context.user_identity,
                    'properties': {**event.properties, 'session': self.context.session_identity},
                    'timestamp': event.timestamp.isoformat(),
                })
                query_user_properties.update(event.user_properties)

            if len(query_events) == 1:
                query_events = query_events[0]
            if not self.__api.track(query_events):
                return

            if query_user_properties:
                query_user: UserProperties = {
                    'identity': self.context.user_identity,
                    'properties': query_user_properties,
                }
                if not self.__api.add_user_properties(query_user):
                    events[:len(query_events)] = [
                        item
                        for item in events[:len(query_events)]
                        if item.user_properties
                    ]
                    return

            events[:len(query_events)] = []
