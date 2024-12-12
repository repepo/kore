# -*- coding: utf-8 -*-

"""Collection of utilities for importing python entities."""

from __future__ import annotations

__all__ = ['import_callable', 'import_class']

import typing
from . import exceptions


TypeT = typing.TypeVar('TypeT', bound=type)


def _import_anything(name: typing.Optional[str]) -> typing.Any:
    """Locate any item by its import path."""
    if name is None:
        raise exceptions.ValidationError('value is required')

    if name.startswith('::') and ('.' not in name):
        name = name[2:]
    else:
        raise exceptions.ValidationError('invalid identifier')

    from . import bundle  # pylint: disable=cyclic-import,import-outside-toplevel
    result = getattr(bundle, name, None)

    if result is None:
        raise exceptions.ValidationError('invalid identifier')

    return result


@typing.overload
def import_callable(
        name: typing.Optional[str],
        *,
        allow_none: typing.Literal[False] = False,
) -> typing.Callable[..., typing.Any]:
    ...


@typing.overload
def import_callable(
        name: typing.Optional[str],
        *,
        allow_none: typing.Literal[True],
) -> typing.Optional[typing.Callable[..., typing.Any]]:
    ...


def import_callable(
        name: typing.Optional[str],
        *,
        allow_none: typing.Literal[True, False] = False,
) -> typing.Optional[typing.Callable[..., typing.Any]]:
    """
    Locate a callable object by its import path.

    :param name: Path to callable to import.

                 E.g. :code:`'json.loads'`
    :param allow_none: Allow `name` to be :code:`None`.

                       If allowed and `name` is None - :code:`None` will be returned.
    """
    if (name is None) and allow_none:
        return None

    result: typing.Any = _import_anything(name)

    if not callable(result):
        raise exceptions.ValidationError('is not a function')

    return result


@typing.overload
def import_class(
        name: typing.Optional[str],
        *,
        allow_none: typing.Literal[False] = False,
        inherited_from: typing.Union[TypeT, typing.Sequence[TypeT]] = (),
) -> TypeT:
    ...


@typing.overload
def import_class(
        name: typing.Optional[str],
        *,
        allow_none: typing.Literal[True],
        inherited_from: typing.Union[TypeT, typing.Sequence[TypeT]] = (),
) -> typing.Optional[TypeT]:
    ...


def import_class(
        name: typing.Optional[str],
        *,
        allow_none: typing.Literal[True, False] = False,
        inherited_from: typing.Union[TypeT, typing.Sequence[TypeT]] = (),
) -> typing.Optional[typing.Type]:
    """
    Locate a class object by its import path

    :param name: Path to callable to import.

                 E.g. :code:`'json.JSONEncoder'`
    :param allow_none: Allow `name` to be :code:`None`.

                       If allowed and `name` is None - :code:`None` will be returned.
    :param inherited_from: Validate that target class is inherited from one of optional bases
    """
    if (name is None) and allow_none:
        return None

    result: typing.Any = _import_anything(name)

    if not isinstance(result, type):
        raise exceptions.ValidationError('is not a class')

    parents: typing.Tuple[type, ...]
    if isinstance(inherited_from, type):
        parents = (inherited_from,)
    else:
        parents = tuple(inherited_from)

    if inherited_from and not issubclass(result, parents):
        raise exceptions.ValidationError('this class can not be used')

    return result
