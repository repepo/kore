# -*- coding: utf-8 -*-

"""Project wide utils related to `subprocess` module."""

from __future__ import annotations

__all__ = ['CREATE_NO_WINDOW']

import sys
import subprocess  # nosec
import typing


CREATE_NO_WINDOW: typing.Final[int] = subprocess.CREATE_NO_WINDOW if (sys.platform == 'win32') else 0  # type: ignore
