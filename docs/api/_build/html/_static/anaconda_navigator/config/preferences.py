# -*- coding: utf-8 -*-

"""
Internal preferences of the Navigator.

This module should contain all preferences for the Navigator components, that are constant for the current Navigator
release.

.. note::

    The primary goal of this file - is to have a single place with configurations, instead of spreading them across the
    whole application. Thus, if we need to change a single URL, period, behavior - we may just look into a single file
    instead of looking across the related components to what should be changed.

.. warning::

    If you need any additional data structure for any preference - put it in the
    :mod:`~anaconda_navigator.config.structures`.

    The :mod:`~anaconda_navigator.config.preferences` should contain only preference, which should make it much easier
    to navigate through the file.
"""

from __future__ import annotations

__all__ = ()

import os
import typing

from anaconda_navigator.static import images
from . import base
from . import structures

if typing.TYPE_CHECKING:
    from anaconda_navigator.widgets.dialogs.login import cloud_dialogs


SECONDS: typing.Final[int] = 1
MINUTES: typing.Final[int] = 60 * SECONDS
HOURS: typing.Final[int] = 60 * MINUTES
DAYS: typing.Final[int] = 24 * HOURS


# ╠════════════════════════════════════════════════════════════════════════════════════════════════════════╡ Conda ╞═══╣

FEATURED_CHANNELS: typing.Final[typing.Sequence[str]] = ()


# ╠══════════════════════════════════════════════════════════════════════════════════════════════════╡ Main Window ╞═══╣

SIDEBAR_LINKS: typing.Final[typing.Sequence[structures.SidebarLink]] = (
    structures.SidebarLink('Documentation', 'https://docs.anaconda.com/free/navigator', utm_medium='nav-docs'),
    structures.SidebarLink('Anaconda Blog', 'https://www.anaconda.com/blog', utm_medium='nav-blog'),
)

SIDEBAR_SOCIALS: typing.Final[typing.Sequence[structures.SidebarSocial]] = (
    structures.SidebarSocial('Twitter', url='https://twitter.com/AnacondaInc'),
    structures.SidebarSocial('Youtube', url='https://www.youtube.com/c/continuumio'),
    structures.SidebarSocial('Github', url='https://github.com/ContinuumIO'),
)


# ╠══════════════════════════════════════════════════════════════════════════════════════════════╡ Welcome sign-in ╞═══╣

CLOUD_METADATA_SOURCE: typing.Final[str | None] = None
CLOUD_DEFAULT_METADATA: typing.Final[cloud_dialogs.Metadata] = {
    'pages': [
        {'background': os.path.join(images.CLOUD_BACKGROUNDS_PATH, 'data.png')},
        {'background': os.path.join(images.CLOUD_BACKGROUNDS_PATH, 'notebooks.png')},
        {'background': os.path.join(images.CLOUD_BACKGROUNDS_PATH, 'publish.png')}
    ],
}
WELCOME_DELAYS: typing.Final[structures.Intervals[int]] = structures.Intervals(
    structures.Interval(count=1, value=5 * DAYS),
    structures.Interval(count=1, value=30 * DAYS),
    offset=1,
)


# ╠═══════════════════════════════════════════════════════════════════════════════════════════════╡ Advertisements ╞═══╣

def __init_ad_configuration_paths() -> typing.MutableSequence[str]:
    """Initialize sequence of paths to search configurations in."""
    result: typing.List[str] = []

    if os.name == 'nt':
        # pylint: disable=import-outside-toplevel
        from anaconda_navigator.external.knownfolders import get_folder_path, FOLDERID  # type: ignore
        result.extend((
            os.path.join(get_folder_path(FOLDERID.ProgramData)[0], 'Anaconda3', 'etc', 'partner.yml'),
            os.path.join(get_folder_path(FOLDERID.ProgramData)[0], 'Miniconda3', 'etc', 'partner.yml'),
            os.path.join(os.path.expanduser('~'), 'Anaconda3', 'etc', 'partner.yml'),
            os.path.join(os.path.expanduser('~'), 'Miniconda3', 'etc', 'partner.yml'),
        ))

    else:
        result.extend((
            os.path.join('/', 'etc', 'anaconda', 'partner.yml'),
            os.path.join('opt', 'anaconda3', 'etc', 'partner.yml'),
            os.path.join('opt', 'miniconda3', 'etc', 'partner.yml'),
            os.path.join(os.path.expanduser('~'), 'anaconda3', 'etc', 'partner.yml'),
            os.path.join(os.path.expanduser('~'), 'miniconda3', 'etc', 'partner.yml'),
        ))

    result.append(base.get_conf_path('partner.yml'))

    return result


AD_CONFIGURATION_PATHS: typing.Final[typing.Sequence[str]] = __init_ad_configuration_paths()
AD_SLIDESHOW_TIMEOUT: typing.Final[int] = 60 * SECONDS
AD_SOURCES: typing.Final[typing.Sequence[str]] = (
    'https://anaconda.cloud/api/billboard/v1/ads/navigator/partner/{partner_identity}',
    'https://anaconda.cloud/api/billboard/v1/ads/navigator',
)
