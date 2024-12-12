# -*- coding: utf8 -*-

"""Notification utilities."""

from __future__ import annotations

__all__ = ('Notification', 'NotificationListener', 'NotificationQueue', 'NOTIFICATION_QUEUE')

import collections.abc
import typing

from qtpy.QtCore import QObject, Signal  # pylint: disable=no-name-in-module

if typing.TYPE_CHECKING:
    import typing_extensions


RawNotification: typing_extensions.TypeAlias = 'Notification | str'
NotificationSource: typing_extensions.TypeAlias = 'RawNotification | collections.abc.Iterable[RawNotification] | None'


class Notification:
    """Details on a notification to show."""

    __slots__ = ('__caption', '__hashsum', '__message', '__tags')

    def __init__(
            self,
            message: str,
            caption: str = 'Notification',
            tags: collections.abc.Iterable[str] | str = (),
    ) -> None:
        """Initialize new instance of :class:`~Notification`."""
        if isinstance(tags, str) or (not isinstance(tags, collections.abc.Iterable)):
            tags = (str(tags),)
        else:
            tags = tuple(sorted(map(str, tags)))

        self.__caption: typing.Final[str] = str(caption)
        self.__message: typing.Final[str] = str(message)
        self.__tags: typing.Final[tuple[str, ...]] = tags
        self.__hashsum: typing.Final[int] = hash((self.__caption, self.__message) + tags)

    @property
    def caption(self) -> str:  # noqa: D401
        """Caption for the notification window."""
        return self.__caption

    @property
    def message(self) -> str:  # noqa: D401
        """Content of the notification."""
        return self.__message

    @property
    def tags(self) -> tuple[str, ...]:  # noqa: D401
        """Extra tags for the notification."""
        return self.__tags

    def __eq__(self, other: typing.Any) -> bool:
        """Check if instance is equal to another one."""
        if isinstance(other, Notification):
            return (self.caption == other.caption) and (self.message == other.message) and (self.tags == other.tags)
        return NotImplemented

    def __hash__(self) -> int:
        """Retrieve hash sum of an instance."""
        return self.__hashsum

    def __repr__(self) -> str:
        """Prepare string representation of an instance."""
        return f'{type(self).__name__}(message={self.__message!r}, caption={self.__caption!r}, tags={self.__tags!r})'

    def __str__(self) -> str:
        """Prepare string representation of an instance."""
        return self.__message


class NotificationCollection(typing.MutableSequence[Notification]):
    """Collection of :class:`~Notification` instances."""

    __slots__ = ('__content',)

    def __init__(self, content: collections.abc.Iterable[Notification] = ()) -> None:
        """Initialize new instance of :class:`~NotificationCollection`."""
        self.__content: typing.Final[list[Notification]] = list(content)

    @staticmethod
    def collect(
            message: NotificationSource,
            caption: str = 'Notification',
            tags: str | collections.abc.Sequence[str] = (),
    ) -> NotificationCollection:
        """
        Collect notifications from a :class:`~Notification` compatible source.

        :param message: Message text or source (existing :class:`~Notification` or list of them).
        :param caption: Default caption to apply to non :class:`~Notification` source items.
        :param tags: Default tags to apply to non :class:`~Notification` source items.
        """
        result: list[Notification] = []

        if message is None:
            message = []
        elif isinstance(message, (Notification, str)) or (not isinstance(message, collections.abc.Iterable)):
            message = [message]

        item: Notification | str
        for item in message:
            if not isinstance(item, Notification):
                item = Notification(caption=caption, message=item, tags=tags)
            result.append(item)

        return NotificationCollection(result)

    def exclude(self, items: collections.abc.Container[Notification]) -> NotificationCollection:
        """Prepare a new :code:`~NotificationCollection` from the current content except :code:`items`."""
        return NotificationCollection(
            item
            for item in self.__content
            if item not in items
        )

    def insert(self, index: int, value: Notification) -> None:
        """Insert new :code:`value` at :code:`index`."""
        self.__content.insert(index, value)

    def only(
            self,
            message: str | None = None,
            caption: str | None = None,
            tags: collections.abc.Iterable[str] | str | None = None,
            strict: bool = False,
    ) -> NotificationCollection:
        """
        Filter current collection to only include notifications with particular properties.

        :param message: Expected value of a :attr:`~Notification.message`.
        :param caption: Expected value of a :attr:`~Notification.caption`.
        :param tags: Expected subset of :attr:`~Notification.tags`.
        :param strict: Use :code:`tags` for a full comparison instead of just subset.
        """
        conditions: list[collections.abc.Callable[[Notification], bool]] = []

        if message is not None:
            conditions.append(lambda item: item.message == message)

        if caption is not None:
            conditions.append(lambda item: item.caption == caption)

        if tags is not None:
            if isinstance(tags, str) or (not isinstance(tags, collections.abc.Iterable)):
                tags = [tags]
            if strict:
                tags = tuple(sorted(map(str, tags)))
                conditions.append(lambda item: item.tags == tags)
            else:
                tags = set(map(str, tags))
                conditions.append(lambda item: typing.cast('set[str]', tags).issubset(item.tags))

        return NotificationCollection(
            item
            for item in self.__content
            if all(condition(item) for condition in conditions)
        )

    def unique(self) -> NotificationCollection:
        """Prepare a new :class:`~NotificationCollection` that only contains unique elements."""
        result: NotificationCollection = NotificationCollection()
        result.extend(item for item in self.__content if item not in result)
        return result

    def __delitem__(self, index: int | slice) -> None:
        """Remove item by its index."""
        del self.__content[index]

    @typing.overload
    def __getitem__(self, index: int) -> Notification:
        """Retrieve content from a collection."""

    @typing.overload
    def __getitem__(self, index: slice) -> NotificationCollection:
        """Retrieve content from a collection."""

    def __getitem__(self, index: int | slice) -> Notification | NotificationCollection:
        """Retrieve content from a collection."""
        if isinstance(index, slice):
            return NotificationCollection(self.__content[index])
        return self.__content[index]

    def __iter__(self) -> collections.abc.Iterator[Notification]:
        """Iterate through the content of a collection."""
        return iter(self.__content)

    def __len__(self) -> int:
        """Retrieve total number of added :class:`~Notification` instances."""
        return len(self.__content)

    def __repr__(self) -> str:
        """Prepare string representation of an instance."""
        return f'{type(self).__name__}({repr(self.__content)})'

    @typing.overload
    def __setitem__(self, index: int, value: Notification) -> None:
        """Update value of an item."""

    @typing.overload
    def __setitem__(self, index: slice, value: collections.abc.Iterable[Notification]) -> None:
        """Update value of an item."""

    def __setitem__(self, index: int | slice, value: Notification | collections.abc.Iterable[Notification]) -> None:
        """Update value of an item."""
        self.__content[index] = value  # type: ignore

    def __str__(self) -> str:
        """Prepare string representation of an instance."""
        return str(self.__content)


class NotificationListener(QObject):
    """Dynamic listener for notifications."""

    sig_notification = Signal(Notification)

    def push(self, notification: Notification) -> None:
        """Push new notification to the listener."""
        self.sig_notification.emit(notification)


class NotificationQueue:
    """Notification dialogs queue."""

    __slots__ = ('__history', '__listener', '__queue')

    def __init__(self) -> None:
        """Initialize new instance of :class:`~NotificationQueue`."""
        self.__history: typing.Final[set[Notification]] = set()
        self.__listener: NotificationListener | None = None
        self.__queue: typing.Final[collections.deque[Notification]] = collections.deque()

    def attach(self, listener: NotificationListener) -> None:
        """
        Attach listener to the queue.

        If there are any notifications pending - all of them will be dumped into the :code:`listener`.
        """
        self.__listener = listener
        while self.__queue:
            self.__listener.push(self.__queue.popleft())

    def push(
            self,
            message: NotificationSource,
            caption: str = 'Notification',
            tags: collections.abc.Sequence[str] | str = (),
            *,
            once: bool = False,
            unique: bool = True,
    ) -> NotificationCollection:
        """
        Push new notification to the queue.

        :param message: Message text or source (existing :class:`~Notification` or list of them).
        :param caption: Default caption to apply to non :class:`~Notification` source items.
        :param tags: Default tags to apply to non :class:`~Notification` source items.
        :param once: Show each notification only once per application session.
        :param unique: Show each notification only once per :meth:`~NotificationQueue.push` request. May also hide
                       duplicates until the listener is attached via :meth:`~NotificationQueue.attach`.
        """
        result: NotificationCollection = NotificationCollection.collect(message=message, caption=caption, tags=tags)
        delta: NotificationCollection = result

        if unique or once:
            delta = delta.unique()
        if once:
            delta = delta.exclude(self.__history)
            self.__history.update(delta)

        if self.__listener is None:
            if unique:
                delta = delta.exclude(self.__queue)
            self.__queue.extend(delta)
        else:
            for raw in delta:
                self.__listener.push(raw)

        return result


NOTIFICATION_QUEUE: typing.Final[NotificationQueue] = NotificationQueue()
