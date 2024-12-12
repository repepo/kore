# -*- coding: utf-8 -*-

"""Collection of utility components to use by APIs."""

from __future__ import annotations

__all__ = ['split_canonical_name']

import typing


def split_canonical_name(cname: str) -> typing.Tuple[str, ...]:
    """Split a canonical package name into name, version, build."""
    return tuple(cname.rsplit('-', 2))
