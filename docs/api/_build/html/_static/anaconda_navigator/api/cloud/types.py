# -*- coding: utf-8 -*-

"""Additional types used by CloudAPI."""

from __future__ import annotations

__all__ = ()

import typing


class CloudEnvironmentRecord(typing.TypedDict):
    """Description of the environment reported by Cloud API."""

    id: str  # pylint: disable=invalid-name
    name: str
    yaml_ref: str
    revision: int
    created_at: str
    updated_at: str


class CloudEnvironmentCollection(typing.TypedDict):
    """Collection of environments reported by Cloud API."""

    items: typing.Sequence[CloudEnvironmentRecord]
    total: int
