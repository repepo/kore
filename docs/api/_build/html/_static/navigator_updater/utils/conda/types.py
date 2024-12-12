# -*- coding: utf-8 -*-

"""Type definitions for the :mod:`~anaconda_navigator.utils.conda`."""

from __future__ import annotations

__all__ = ()

import typing


CondaInfoOutput = typing.TypedDict('CondaInfoOutput', {
    'GID': int,
    'UID': int,
    'active_prefix': typing.Optional[str],
    'active_prefix_name': typing.Optional[str],
    'av_data_dir': str,
    'av_metadata_url_base': typing.Optional[str],
    'channels': typing.List[str],
    'conda_build_version': str,
    'conda_env_version': str,
    'conda_location': str,
    'conda_prefix': str,
    'conda_shlvl': int,
    'conda_version': str,
    'config_files': typing.List[str],
    'default_prefix': str,
    'env_vars': dict,
    'envs': typing.List[str],
    'envs_dirs': typing.List[str],
    'netrc_file': typing.Optional[str],
    'offline': bool,
    'pkgs_dirs': typing.List[str],
    'platform': str,
    'python_version': str,
    'rc_path': str,
    'requests_version': str,
    'root_prefix': str,
    'root_writable': bool,
    'site_dirs': typing.List[str],
    'sys.executable': str,
    'sys.prefix': str,
    'sys.version': str,
    'sys_rc_path': str,
    'user_agent': str,
    'user_rc_path': str,
    'virtual_pkgs': typing.List[typing.List[str]],
})


class CondaErrorOutput(typing.TypedDict):
    """Common structure of the error reported by Conda."""

    caused_by: str
    error: str
    exception_name: str
    exception_type: str
    message: str


class CondaValidationErrorOutput(CondaErrorOutput):
    """Error body of the :code:`CustomValidationError` reported by Conda."""

    parameter_name: str
    parameter_value: str
    source: str
