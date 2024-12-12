# -*- coding: utf-8 -*-

"""Collection of utility components to use by APIs."""

from __future__ import annotations

__all__ = ['normalize_certificate', 'is_internet_available', 'split_canonical_name']

import configparser
import os
import typing

from anaconda_navigator.config import CONF


def normalize_certificate(value: typing.Union[None, bool, str]) -> typing.Union[None, bool, str]:
    """Check if certificate value is valid and fix it if required."""
    if isinstance(value, str) and (not os.path.exists(value)):
        return True
    return value


def is_internet_available() -> bool:
    """Check internet availability."""
    try:
        config_value = CONF.get('main', 'offline_mode')
    except (AttributeError, configparser.NoOptionError):
        return True
    return not bool(config_value)


def split_canonical_name(cname: str) -> typing.Tuple[str, ...]:
    """Split a canonical package name into name, version, build."""
    return tuple(cname.rsplit('-', 2))
