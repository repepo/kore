# -*- coding: utf-8 -*-

"""Utilities for common validation tasks."""

from __future__ import annotations

__all__ = [
    # helper context managers
    'catch_exception',

    # helper iterators
    'iterable_items', 'mapping_items',

    # helper functions
    'get_mapping_item', 'pop_mapping_item',

    # helper checkers
    'all_of', 'any_of',

    # value checkers
    'each_item', 'each_mapping_value',
    'of_options', 'of_type', 'has_items', 'mapping_is_empty',

    # shortcuts
    'is_str', 'is_str_or_none',
    'is_int', 'is_int_or_none',
    'is_bool', 'is_bool_or_none',
    'is_str_sequence', 'is_int_sequence',
]

import contextlib
import typing
from . import exceptions


T = typing.TypeVar('T')
KeyT = typing.TypeVar('KeyT', str, int)

EMPTY: typing.Any = object()


class ValueChecker(typing.Protocol):  # pylint: disable=too-few-public-methods
    """Common interface for functions that can be used as a value validators."""

    def __call__(
            self,
            value: typing.Any,
            field_name: typing.Union[str, int, typing.Iterable[typing.Union[str, int]]] = (),
    ) -> None:
        """
        Check if `value` is valid.

        :param value: Value to check.
        :param field_name: Name of the field, for which validation error should be reported.
        """


# ╠══════════════════════════════════════════════════════════════════════════════════════╡ helper context managers ╞═══╣

@contextlib.contextmanager
def catch_exception(
        exception_type: typing.Union[
            typing.Type[BaseException],
            typing.Tuple[typing.Type[BaseException], ...],
        ] = BaseException,
        message: typing.Optional[str] = None,
        field_name: typing.Union[str, int, typing.Iterable[typing.Union[str, int]]] = (),
) -> typing.Iterator[None]:
    """
    Convert any :exc:`~Exception` to :exc:`~anaconda_navigator.api.external_apps.exceptions.ValidationError`.

    :param exception_type: Convert only this type(s) of :exc:`~BaseException` instances.
    :param message: Custom message to append.

                    If not provided - string representation of original :exc:`~BaseException` will be used.
    :param field_name: Report exception for a particular field.

    Examples:

    >>> with catch_exception():
    ...     _ = 1 / 0
    Traceback (most recent call last):
        ...
    anaconda_navigator.api.external_apps.exceptions.ValidationError: division by zero

    >>> with catch_exception(ValueError):
    ...     _ = 1 / 0
    Traceback (most recent call last):
        ...
    ZeroDivisionError: division by zero
    """
    try:
        yield
    except exception_type as error:
        custom_message: str
        if message is not None:
            custom_message = message
        elif isinstance(error, exceptions.ValidationError):
            custom_message = error.message
        else:
            custom_message = str(error)

        raise exceptions.ValidationError(custom_message, field=field_name) from error


# ╠═════════════════════════════════════════════════════════════════════════════════════════════╡ helper iterators ╞═══╣

def iterable_items(source: typing.Iterable[T]) -> typing.Iterator[T]:
    """
    Iterate through items of any iterable.

    For each item - it's index will be used as the name of a field being validated.

    Example:

    >>> with exceptions.ValidationError.with_field('list'):
    ...     for item in iterable_items([1, 2, 'text', 3]):
    ...         if not isinstance(item, int):
    ...             raise exceptions.ValidationError(message='wrong type', field='value')
    Traceback (most recent call last):
        ...
    anaconda_navigator.api.external_apps.exceptions.ValidationError: list.2.value: wrong type
    """
    index: int
    value: T
    for index, value in enumerate(source):
        with exceptions.ValidationError.with_field(index):
            yield value


def mapping_items(source: typing.Mapping[KeyT, T]) -> typing.Iterator[typing.Tuple[KeyT, T]]:
    """
    Iterate through item tuples of a mapping

    For each such tuple - key of the item will be used as the name of a field being validated.

    Example:

    >>> with exceptions.ValidationError.with_field('dict'):
    ...     for k, v in mapping_items({'field_a': 1, 'field_b': 2, 'field_c': 'text', 'field_d': 3}):
    ...         if not isinstance(v, int):
    ...             raise exceptions.ValidationError(message='wrong type', field='value')
    Traceback (most recent call last):
        ...
    anaconda_navigator.api.external_apps.exceptions.ValidationError: dict.field_c.value: wrong type
    """
    key: KeyT
    value: T
    for key, value in source.items():
        with exceptions.ValidationError.with_field(key):
            yield key, value


# ╠═════════════════════════════════════════════════════════════════════════════════════════════╡ helper functions ╞═══╣

def get_mapping_item(
        mapping: typing.Mapping[KeyT, typing.Any],
        item: KeyT,
        default: typing.Any = EMPTY,
) -> typing.Any:
    """
    Get `item` from any `mapping`.

    If `item` not found and `default` is not provided -
    :exc:`~anaconda_navigator.api.external_apps.exceptions.ValidationError` will be raised.
    """
    try:
        return mapping[item]
    except KeyError:
        if default is EMPTY:
            raise exceptions.ValidationError(f'{item!r} must be provided') from None
        return default


def pop_mapping_item(
        mapping: typing.MutableMapping[KeyT, typing.Any],
        item: KeyT,
        default: typing.Any = EMPTY,
) -> typing.Any:
    """
    Pop `item` from any `mapping`.

    If `item` not found and `default` is not provided -
    :exc:`~anaconda_navigator.api.external_apps.exceptions.ValidationError` will be raised.
    """
    try:
        return mapping.pop(item)
    except KeyError:
        if default is EMPTY:
            raise exceptions.ValidationError(f'{item!r} must be provided') from None
        return default


# ╠══════════════════════════════════════════════════════════════════════════════════════════════╡ helper checkers ╞═══╣

def all_of(*args: 'ValueChecker') -> 'ValueChecker':
    """Check that value passes all checks."""
    def checker(
            value: typing.Any,
            field_name: typing.Union[str, int, typing.Iterable[typing.Union[str, int]]] = (),
    ) -> None:
        """
        Check if `value` is valid.

        :param value: Value to check.
        :param field_name: Name of the field, for which validation error should be reported.
        """
        arg: 'ValueChecker'
        for arg in args:
            arg(value, field_name=field_name)

    return checker


def any_of(*args: 'ValueChecker') -> 'ValueChecker':
    """Check that value passes at least one check."""
    def checker(
            value: typing.Any,
            field_name: typing.Union[str, int, typing.Iterable[typing.Union[str, int]]] = (),
    ) -> None:
        """
        Check if `value` is valid.

        :param value: Value to check.
        :param field_name: Name of the field, for which validation error should be reported.
        """
        arg: 'ValueChecker'
        iterator: typing.Iterator['ValueChecker'] = iter(args)
        try:
            for arg in iterator:
                arg(value, field_name=field_name)
                break
        except exceptions.ValidationError:
            for arg in iterator:
                with contextlib.suppress(exceptions.ValidationError):
                    arg(value, field_name=field_name)
                    break
            else:
                raise

    return checker


# ╠═══════════════════════════════════════════════════════════════════════════════════════════════╡ value checkers ╞═══╣

def each_item(*args: 'ValueChecker') -> 'ValueChecker':
    """
    Validate each item of any iterable.

    Example:

    >>> validator = each_item(of_type(int))
    >>> validator([1, 2, 3])
    >>> validator([1, 2, 'text', 3])
    Traceback (most recent call last):
        ...
    anaconda_navigator.api.external_apps.exceptions.ValidationError: 2: must be a int, not str

    >>> validator([1, 2, 'text', 3], field_name='validated_item')
    Traceback (most recent call last):
        ...
    anaconda_navigator.api.external_apps.exceptions.ValidationError: validated_item.2: must be a int, not str
    """
    def checker(
            value: typing.Any,
            field_name: typing.Union[str, int, typing.Iterable[typing.Union[str, int]]] = (),
    ) -> None:
        """
        Check if `value` is valid.

        :param value: Value to check.
        :param field_name: Name of the field, for which validation error should be reported.
        """
        with exceptions.ValidationError.with_field(field_name):
            item: typing.Any
            for item in iterable_items(value):
                arg: 'ValueChecker'
                for arg in args:
                    arg(item)

    return checker


def each_mapping_value(*args: 'ValueChecker') -> 'ValueChecker':
    """
    Validate each value in mapping.

    Example:

    >>> validator = each_mapping_value(of_type(int))
    >>> validator({'first': 1, 'second': 2, 'third': 3})
    >>> validator({'first': 1, 'second': 2, 'third': 'text', 'fourth': 3})
    Traceback (most recent call last):
        ...
    anaconda_navigator.api.external_apps.exceptions.ValidationError: third: must be a int, not str

    >>> validator({'first': 1, 'second': 2, 'third': 'text', 'fourth': 3}, field_name='validated_item')
    Traceback (most recent call last):
        ...
    anaconda_navigator.api.external_apps.exceptions.ValidationError: validated_item.third: must be a int, not str
    """
    def checker(
            value: typing.Any,
            field_name: typing.Union[str, int, typing.Iterable[typing.Union[str, int]]] = (),
    ) -> None:
        """
        Check if `value` is valid.

        :param value: Value to check.
        :param field_name: Name of the field, for which validation error should be reported.
        """
        with exceptions.ValidationError.with_field(field_name):
            item: typing.Any
            for _, item in mapping_items(value):
                arg: 'ValueChecker'
                for arg in args:
                    arg(item)

    return checker


def of_options(
        *options: typing.Any,
        allow_none: bool = False,
) -> 'ValueChecker':
    """
    Validate that value is one of the options.

    Examples:

    >>> validator = of_options('a', 'b', 'c')
    >>> validator('b')
    >>> validator('c')
    >>> validator('d')
    Traceback (most recent call last):
        ...
    anaconda_navigator.api.external_apps.exceptions.ValidationError: must be one of 'a', 'b', 'c', not 'd'
    """
    def checker(
            value: typing.Any,
            field_name: typing.Union[str, int, typing.Iterable[typing.Union[str, int]]] = (),
    ) -> None:
        """
        Check if `value` is valid.

        :param value: Value to check.
        :param field_name: Name of the field, for which validation error should be reported.
        """
        if allow_none and (value is None):
            return

        if value in options:
            return

        message: str = ', '.join(repr(item) for item in options)
        raise exceptions.ValidationError(f'must be one of {message}, not {value!r}', field=field_name)

    return checker


def of_type(
        *types: typing.Type,
        allow_none: bool = False,
) -> 'ValueChecker':
    """
    Validate type of the value.

    Examples:

    >>> validator = of_type(int)
    >>> validator(1)
    >>> validator(None)
    Traceback (most recent call last):
        ...
    anaconda_navigator.api.external_apps.exceptions.ValidationError: must be a int, not NoneType

    >>> validator('text', field_name='value')
    Traceback (most recent call last):
        ...
    anaconda_navigator.api.external_apps.exceptions.ValidationError: value: must be a int, not str

    >>> validator = of_type(int, allow_none=True)
    >>> validator(1)
    >>> validator(None)
    >>> validator('text')
    Traceback (most recent call last):
        ...
    anaconda_navigator.api.external_apps.exceptions.ValidationError: must be a int, not str
    """
    def checker(
            value: typing.Any,
            field_name: typing.Union[str, int, typing.Iterable[typing.Union[str, int]]] = (),
    ) -> None:
        """
        Check if `value` is valid.

        :param value: Value to check.
        :param field_name: Name of the field, for which validation error should be reported.
        """
        if allow_none and (value is None):
            return

        if isinstance(value, types):
            return

        message: str = ' | '.join(getattr(item, '__name__', str(item)) for item in types)
        raise exceptions.ValidationError(f'must be a {message}, not {type(value).__name__}', field=field_name)

    return checker


def has_items(
        *,
        at_least: typing.Optional[int] = None,
        at_most: typing.Optional[int] = None,
) -> 'ValueChecker':
    """
    Validate that collection contains `at_least` and `at_most` items.

    Examples:

    >>> validator = has_items(at_least=1)
    >>> validator([1])
    >>> validator([])
    Traceback (most recent call last):
        ...
    anaconda_navigator.api.external_apps.exceptions.ValidationError: must have at least 1 items

    >>> validator = has_items(at_most=1)
    >>> validator([])
    >>> validator([1])
    >>> validator([1, 2])
    Traceback (most recent call last):
        ...
    anaconda_navigator.api.external_apps.exceptions.ValidationError: must have at most 1 items

    >>> validator = has_items(at_least=1, at_most=2)
    >>> validator([1])
    >>> validator([1, 2])
    >>> validator([1, 2, 3])
    Traceback (most recent call last):
        ...
    anaconda_navigator.api.external_apps.exceptions.ValidationError: must have at most 2 items
    """
    def checker(
            value: typing.Any,
            field_name: typing.Union[str, int, typing.Iterable[typing.Union[str, int]]] = (),
    ) -> None:
        """
        Check if `value` is valid.

        :param value: Value to check.
        :param field_name: Name of the field, for which validation error should be reported.
        """
        if (at_least is not None) and (len(value) < at_least):
            raise exceptions.ValidationError(f'must have at least {at_least} items', field=field_name)
        if (at_most is not None) and (len(value) > at_most):
            raise exceptions.ValidationError(f'must have at most {at_most} items', field=field_name)

    return checker


def mapping_is_empty() -> 'ValueChecker':
    """
    Validate that mapping has no items in it.

    Examples:

    >>> validator = mapping_is_empty()
    >>> validator({})
    >>> validator({'first': 'value', 'second': 'value'})
    Traceback (most recent call last):
        ...
    anaconda_navigator.api.external_apps.exceptions.ValidationError: unexpected items: 'first', 'second'
    """
    def checker(
            value: typing.Any,
            field_name: typing.Union[str, int, typing.Iterable[typing.Union[str, int]]] = (),
    ) -> None:
        """
        Check if `value` is valid.

        :param value: Value to check.
        :param field_name: Name of the field, for which validation error should be reported.
        """
        if value:
            raise exceptions.ValidationError(
                f'unexpected items: {", ".join(repr(item) for item in value)}',
                field=field_name,
            )

    return checker


# ╠════════════════════════════════════════════════════════════════════════════════════════════════════╡ shortcuts ╞═══╣
# pylint: disable=invalid-name

is_str: typing.Final[ValueChecker] = of_type(str)
is_str_or_none: typing.Final[ValueChecker] = of_type(str, allow_none=True)

is_int: typing.Final[ValueChecker] = of_type(int)
is_int_or_none: typing.Final[ValueChecker] = of_type(int, allow_none=True)

is_bool: typing.Final[ValueChecker] = of_type(bool)
is_bool_or_none: typing.Final[ValueChecker] = of_type(bool, allow_none=True)

is_str_sequence: typing.Final[ValueChecker] = all_of(of_type(typing.Sequence), each_item(is_str))
is_int_sequence: typing.Final[ValueChecker] = all_of(of_type(typing.Sequence), each_item(is_int))
