# -*- coding: utf-8 -*-

"""Collection of additional helper utilities for dialog management."""

__all__ = ['WidgetGroup']

import typing
from qtpy import QtWidgets


class WidgetGroup(typing.Sequence[QtWidgets.QWidget]):
    """Collection of widgets, with ability for synchronous control."""

    __slots__ = ('__content',)

    def __init__(self, *args: typing.Union[QtWidgets.QWidget, typing.Iterable[QtWidgets.QWidget]]) -> None:
        """Initialize new :class:`~WidgetGroup` instance."""
        self.__content: typing.Final[typing.List[QtWidgets.QWidget]] = []

        arg: typing.Union[QtWidgets.QWidget, typing.Iterable[QtWidgets.QWidget]]
        for arg in args:
            if isinstance(arg, QtWidgets.QWidget):
                self.__content.append(arg)
            else:
                self.__content.extend(arg)

    def disable(self, state: bool = True) -> None:
        """Set disabled state to all widgets in the group."""
        self.for_each(action=lambda item: item.setDisabled(state))

    def enable(self, state: bool = True) -> None:
        """Set enabled state to all widgets in the group."""
        self.for_each(action=lambda item: item.setEnabled(state))

    def for_each(self, action: typing.Callable[[QtWidgets.QWidget], typing.Any]) -> None:
        """Preform same `action` for each widget in the group."""
        item: 'QtWidgets.QWidget'
        for item in self.__content:
            action(item)

    def hide(self, state: bool = True) -> None:
        """Set hidden state to all widgets in the group."""
        self.for_each(action=lambda item: item.setHidden(state))

    def only(self, predicate: typing.Callable[[QtWidgets.QWidget], bool]) -> 'WidgetGroup':
        """Filter widgets in group by predicate."""
        return WidgetGroup(
            item
            for item in self.__content
            if predicate(item)
        )

    def only_disabled(self) -> 'WidgetGroup':
        """Select widgets that are disabled."""
        return self.only(predicate=lambda item: not item.isEnabled())

    def only_enabled(self) -> 'WidgetGroup':
        """Select widgets that are enabled."""
        return self.only(predicate=lambda item: item.isEnabled())

    def only_hidden(self) -> 'WidgetGroup':
        """Select widgets that are hidden."""
        return self.only(predicate=lambda item: not item.isVisible())

    def only_visible(self) -> 'WidgetGroup':
        """Select widgets that are visible."""
        return self.only(predicate=lambda item: item.isVisible())

    def show(self, state: bool = True) -> None:
        """Set visible state to all widgets in cgroup."""
        self.for_each(action=lambda item: item.setVisible(state))

    def __add__(self, other: 'WidgetGroup') -> 'WidgetGroup':
        """Concatenate two widget groups."""
        if not isinstance(other, WidgetGroup):
            return NotImplemented  # type: ignore
        return WidgetGroup(self, other)

    @typing.overload
    def __getitem__(self, index: int) -> 'QtWidgets.QWidget':
        """Retrieve single widget by its index."""

    @typing.overload
    def __getitem__(self, index: slice) -> 'WidgetGroup':
        """Retrieve subset of widgets."""

    def __getitem__(self, index: typing.Union[int, slice]) -> typing.Union['QtWidgets.QWidget', 'WidgetGroup']:
        """Retrieve content of the collection."""
        if isinstance(index, int):
            return self.__content[index]
        return WidgetGroup(self.__content[index])

    def __len__(self) -> int:
        """Retrieve total number of items in the collection."""
        return len(self.__content)
