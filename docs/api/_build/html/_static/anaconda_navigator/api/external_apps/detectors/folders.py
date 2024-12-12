# -*- coding: utf-8 -*-

"""Collection of known folders."""

from __future__ import annotations

__all__ = ['FOLDERS']

import os
import typing
from anaconda_navigator import config


Folder = typing.Literal[
    'linux/root',
    'linux/home',
    'linux/snap_primary',
    'linux/snap_secondary',

    'osx/root',
    'osx/home',
    'osx/applications',
    'osx/user_applications',

    'windows/home',
    'windows/program_files_x86',
    'windows/program_files_x64',
    'windows/program_data',
    'windows/local_app_data',
    'windows/local_app_data_low',
    'windows/roaming_app_data',
]

FOLDER_KEYS: typing.Final[typing.Tuple[Folder, ...]] = Folder.__args__  # type: ignore


def __init_folders() -> typing.MutableMapping[Folder, typing.Optional[str]]:
    """Initialize new mapping with available folder paths."""
    result: typing.MutableMapping[Folder, typing.Optional[str]] = {
        key: None
        for key in FOLDER_KEYS
    }

    root: str
    home: str
    if config.LINUX:
        root = result['linux/root'] = '/'
        home = result['linux/home'] = os.path.expanduser('~')
        result['linux/snap_primary'] = os.path.join(root, 'snap')
        result['linux/snap_secondary'] = os.path.join(home, 'var', 'lib', 'snapd', 'snap')

    if config.MAC:
        root = result['osx/root'] = '/'
        home = result['osx/home'] = os.path.expanduser('~')
        result['osx/applications'] = os.path.join(root, 'Applications')
        result['osx/user_applications'] = os.path.join(home, 'Applications')

    if config.WIN:
        # pylint: disable=import-outside-toplevel
        from anaconda_navigator.external.knownfolders import get_folder_path, FOLDERID  # type: ignore
        result['windows/home'] = os.path.expanduser('~')
        result['windows/program_files_x86'] = get_folder_path(FOLDERID.ProgramFilesX86)[0]
        result['windows/program_files_x64'] = get_folder_path(FOLDERID.ProgramFilesX64)[0]
        result['windows/program_data'] = get_folder_path(FOLDERID.ProgramData)[0]
        result['windows/local_app_data'] = get_folder_path(FOLDERID.LocalAppData)[0]
        result['windows/local_app_data_low'] = get_folder_path(FOLDERID.LocalAppDataLow)[0]
        result['windows/roaming_app_data'] = get_folder_path(FOLDERID.RoamingAppData)[0]

    return result


FOLDERS: typing.Final[typing.Mapping[Folder, typing.Optional[str]]] = __init_folders()
