# -*- coding: utf8 -*-

"""Primary bases for telemetry processing."""

from __future__ import annotations

__all__ = ['Event', 'Context', 'Provider', 'Pool', 'Analytics']

import abc
import collections.abc
import datetime
import queue
import typing
import uuid

from anaconda_navigator.utils import workers

if typing.TYPE_CHECKING:
    import typing_extensions


KeyT = typing.TypeVar('KeyT', bound=collections.abc.Hashable)
ValueT = typing.TypeVar('ValueT')

EventProperty: typing_extensions.TypeAlias = 'str | int'
EventPropertyCollection: typing_extensions.TypeAlias = 'collections.abc.Mapping[str, EventProperty]'
MappingSource: typing_extensions.TypeAlias = (
    'collections.abc.Mapping[KeyT, ValueT] | collections.abc.Iterable[tuple[KeyT, ValueT]]'
)


def generate_identifier() -> str:
    """Generate random identifier string."""
    return f'{uuid.uuid4().hex}{uuid.uuid4().hex}'


class Event:
    """Telemetry event to track."""

    __slots__ = ('__key', '__name', '__properties', '__timestamp', '__user_properties')

    def __init__(
            self,
            name: str,
            properties: MappingSource[str, EventProperty] = (),  # pylint: disable=invalid-sequence-index
            *,
            key: str | None = None,
            timestamp: datetime.datetime | None = None,
            user_properties: MappingSource[str, EventProperty] = (),  # pylint: disable=invalid-sequence-index
    ) -> None:
        """Initialize new instance of a :class:`~Event`."""
        self.__key: typing.Final[str] = str(key or uuid.uuid4().hex)
        self.__name: typing.Final[str] = str(name)
        self.__properties: typing.Final[EventPropertyCollection] = dict(properties)
        self.__timestamp: typing.Final[datetime.datetime] = timestamp or datetime.datetime.now(datetime.timezone.utc)
        self.__user_properties: typing.Final[EventPropertyCollection] = dict(user_properties)

    @property
    def key(self) -> str:  # noqa: D401
        """Unique identifier of the event."""
        return self.__key

    @property
    def name(self) -> str:  # noqa: D401
        """Name of the event."""
        return self.__name

    @property
    def properties(self) -> EventPropertyCollection:  # noqa: D401
        """Extra properties of the event."""
        return self.__properties

    @property
    def timestamp(self) -> datetime.datetime:
        """When event is happened."""
        return self.__timestamp

    @property
    def user_properties(self) -> EventPropertyCollection:  # noqa: D401
        """Extra properties of the user triggered the event."""
        return self.__user_properties


class Context:
    """
    Common context for telemetry processing.

    :param user_identity: Unique identifier of the current user.
    :param session_identity: Unique identifier of the current session.
    """

    __slots__ = ('__session_identity', '__user_identity')

    def __init__(
            self,
            session_identity: str | None = None,
            user_identity: str | None = None,
    ) -> None:
        """Initialize new instance of a :class:`~Context`."""
        if not session_identity:
            session_identity = generate_identifier()
        if not user_identity:
            user_identity = generate_identifier()

        self.__session_identity: typing.Final[str] = str(session_identity)
        self.__user_identity: typing.Final[str] = str(user_identity)

    @property
    def session_identity(self) -> str:  # noqa: D401
        """Unique identifier of the current session."""
        return self.__session_identity

    @property
    def user_identity(self) -> str:  # noqa: D401
        """Unique identifier of the current user."""
        return self.__user_identity


class Provider(metaclass=abc.ABCMeta):
    """
    Integration with telemetry service.

    :param alias: Unique name of the provider.

                  Used to identify providers when events are sent to them.

                  If multiple providers use the same name - each event will be processed by at most one of them. This
                  might be useful for chain of responsibility implementation, but otherwise be careful not to lose
                  collected data.

    :param context: Context with primary environment details.
    """

    __slots__ = ('__alias', '__context')

    def __init__(self, alias: str, context: Context) -> None:
        """Initialize new instance of a :class:`Provider`."""
        self.__alias: typing.Final[str] = alias
        self.__context: typing.Final[Context] = context

    @property
    def alias(self) -> str:  # noqa: D401
        """Unique name of the provider."""
        return self.__alias

    @property
    def context(self) -> Context:  # noqa: D401
        """Context with primary environment details."""
        return self.__context

    @abc.abstractmethod
    def close(self) -> None:
        """Close provider."""

    @abc.abstractmethod
    def process(self, events: collections.abc.MutableSequence[Event]) -> None:
        """
        Process all pending :code:`events`.

        Processed events must be removed from :code:`events`.
        """


class Pool(metaclass=abc.ABCMeta):
    """Storage and manager for telemetry events."""

    __slots__ = ()

    @abc.abstractmethod
    def pending(self, provider: str) -> collections.abc.MutableSequence[Event]:
        """
        Retrieve all currently pending events for :code:`provider`.

        It must be a mutable sequence that provider can modify, and those changes must be applied to the pool. It is
        required for providers to remove events that are already processed and keep only those that are yet to process.
        """

    @abc.abstractmethod
    def push(self, event: Event) -> None:
        """Push new :code:`event` to the pool."""

    @abc.abstractmethod
    def register(self, providers: collections.abc.Iterable[str]) -> None:
        """Register :code:`providers` in the pool for it to be aware of them."""


class Analytics:
    """
    Core component for telemetry management.

    :param providers: Provides to process telemetry with.
    :param pool: Storage for telemetry events.
    """

    __slots__ = ('__pool', '__providers', '__queue')

    def __init__(
            self,
            providers: collections.abc.Iterable[Provider],
            pool: Pool,
    ) -> None:
        """Initialize new instance of an :class:`~Analytics`."""
        self.__pool: typing.Final[Pool] = pool
        self.__providers: typing.Final[collections.abc.Sequence[Provider]] = tuple(providers)
        self.__queue: typing.Final[queue.Queue[Event | None]] = queue.Queue()

        self.__pool.register(provider.alias for provider in self.__providers)

        workers.Task(self.__run, workers.AddCancelCallback(self.stop)).worker().start()

    @property
    def providers(self) -> collections.abc.Sequence[Provider]:  # noqa: D401
        """List of provides used to process events."""
        return self.__providers

    @typing.overload
    def event(self, name: Event) -> None:
        """Push new telemetry event."""

    @typing.overload
    def event(
            self,
            name: str,
            properties: MappingSource[str, EventProperty] = (),  # pylint: disable=invalid-sequence-index
            *,
            key: str | None = None,
            timestamp: datetime.datetime | None = None,
            user_properties: MappingSource[str, EventProperty] = (),  # pylint: disable=invalid-sequence-index
    ) -> None:
        """Push new telemetry event."""

    def event(
            self,
            name: str | Event,
            properties: MappingSource[str, EventProperty] = (),  # pylint: disable=invalid-sequence-index
            *,
            key: str | None = None,
            timestamp: datetime.datetime | None = None,
            user_properties: MappingSource[str, EventProperty] = (),  # pylint: disable=invalid-sequence-index
    ) -> None:
        """Push new telemetry event."""
        if not isinstance(name, Event):
            name = Event(
                key=key,
                name=name,
                properties=properties,
                timestamp=timestamp,
                user_properties=user_properties,
            )

        self.__queue.put(name)

    def stop(self) -> None:
        """Stop processing telemetry."""
        self.__queue.put(None)

    def __run(self) -> None:
        """Primary thread used to process analytics."""
        event: Event | None
        provider: Provider
        while event := self.__queue.get():
            self.__pool.push(event)
            for provider in self.providers:
                provider.process(self.__pool.pending(provider.alias))

        for provider in self.providers:
            provider.close()
