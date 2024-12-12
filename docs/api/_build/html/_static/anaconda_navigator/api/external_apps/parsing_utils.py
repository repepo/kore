# -*- coding: utf-8 -*-

"""Additional utilities used for application parsing."""

from __future__ import annotations

__all__ = ['ParsingContext']

import typing

if typing.TYPE_CHECKING:
    from anaconda_navigator.api import process
    from anaconda_navigator.config import user as user_config


class ParsingContext(typing.NamedTuple):
    """Context used for application parsing."""

    process_api: 'process.WorkerManager'
    user_configuration: 'user_config.UserConfig'
