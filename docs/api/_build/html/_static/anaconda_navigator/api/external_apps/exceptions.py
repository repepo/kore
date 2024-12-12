# -*- coding: utf-8 -*-

"""Custom exceptions for application detectors."""

from __future__ import annotations

__all__ = ['ValidationError']

import contextlib
import typing


def field_tuple(
        value: typing.Union[str, int, typing.Iterable[typing.Union[str, int]]],
) -> typing.Tuple[typing.Union[str, int], ...]:
    """
    Convert field name to uniform format.

    Field names might be of next formats:
    - string - key of dictionary item
    - integer - index of sequence item
    - sequence of any combination of string/integer - multi-level key for multi-level data structures

    Uniform format - is the tuple of any combination of string/integer.
    """
    if isinstance(value, (str, int)):
        return (value,)
    return tuple(value)


class ValidationError(Exception):
    """
    Value didn't pass the validation test.

    :param message: Detailed message about the issue.
    :param field: Path to the field with invalid value.

                  On formats: see :func:`~field_tuple` description.
    :param ignore_stack: Treat `field` as the actual final value.

                         Otherwise - all fields registered with :meth:`~ValidationError.with_field` would be prepended
                         to the provided value.

    Example:

    >>> ValidationError(field=['first', 'second']).field
    ('first', 'second')

    >>> with ValidationError.with_field(['first', 1]):
    ...     ValidationError(field='second').field
    ('first', 1, 'second')
    """

    __stack__: typing.ClassVar[typing.Tuple[typing.Union[str, int], ...]] = ()

    def __init__(
            self,
            message: str = '',
            field: typing.Union[str, int, typing.Iterable[typing.Union[str, int]]] = (),
            ignore_stack: bool = False,
    ) -> None:
        """Initialize new :class:`~ValidationError` instance."""
        super().__init__(message)

        actual_field: typing.Tuple[typing.Union[str, int], ...] = field_tuple(field)
        if not ignore_stack:
            actual_field = self.__stack__ + actual_field

        self.__message: typing.Final[str] = message
        self.__field: typing.Final[typing.Tuple[typing.Union[str, int], ...]] = actual_field

    @classmethod
    @contextlib.contextmanager
    def with_field(cls, name: typing.Union[str, int, typing.Iterable[typing.Union[str, int]]]) -> typing.Iterator[None]:
        """
        Mark, that field with the `name` is being validated.

        :param name: Name of the field being validated.

                     On formats: see :func:`~field_tuple` description.

        Examples:

        >>> with ValidationError.with_field('external'):
        ...     raise ValidationError(message='message')
        Traceback (most recent call last):
            ...
        anaconda_navigator.api.external_apps.exceptions.ValidationError: external: message

        >>> with ValidationError.with_field('external'):
        ...     raise ValidationError(message='message', field='internal')
        Traceback (most recent call last):
            ...
        anaconda_navigator.api.external_apps.exceptions.ValidationError: external.internal: message
        """
        size: int = len(cls.__stack__)
        cls.__stack__ += field_tuple(name)
        try:
            yield
        finally:
            cls.__stack__ = cls.__stack__[:size]

    @property
    def details(self) -> str:  # noqa: D401
        """Verbose details about detected issue."""
        message: str = self.__message or 'invalid value'

        field: str = '.'.join(map(str, self.__field))
        if field:
            return f'{field}: {message}'

        return message

    @property
    def field(self) -> typing.Tuple[typing.Union[int, str], ...]:  # noqa: D401
        """Normalized value of a field name."""
        return self.__field

    @property
    def message(self) -> str:  # noqa: D401
        """Message about issue."""
        return self.__message

    def __repr__(self) -> str:
        """Retrieve string representation of the instance."""
        return f'{type(self).__name__}: {self.details}'

    def __str__(self) -> str:
        """Retrieve string representation of the instance."""
        return self.details
