# -*- coding: utf-8 -*-

"""Exceptions that could be raised by :mod:`~anaconda-navigator.utils.conda` utilities."""

from __future__ import annotations

__all__ = ['CondaError']

import typing

from . import types as conda_types


class CondaError(BaseException):
    """
    General error raised by the Conda process.

    :param error: Content of the Conda response.
    """

    def __init__(self, error: 'conda_types.CondaErrorOutput') -> None:
        """Initialize new :class:`~CondaError` instance."""
        super().__init__()
        self.__error: typing.Final[conda_types.CondaErrorOutput] = error

    @property
    def error(self) -> 'conda_types.CondaErrorOutput':
        """Content of the Conda error output."""
        return self.__error
