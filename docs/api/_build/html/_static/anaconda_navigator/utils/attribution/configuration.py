# -*- coding: utf-8 -*-

"""Additional configuration properties for the attribution."""

from __future__ import annotations

__all__ = ['APPLICABLE_HOSTS']

import typing


APPLICABLE_HOSTS: typing.Final[typing.Sequence[str]] = (
    'anaconda.*',
    '*.anaconda.*',
)
