"""
Configure Conda to use Anaconda Commercial Edition.
"""

import os
import sys
import warnings
from os.path import abspath, expanduser, join

import conda
import conda.gateways.logging  # noqa: F401
import requests
from conda.base.context import reset_context
from conda.cli.python_api import Commands, run_command
from conda.exceptions import CondaKeyError
from conda.gateways.anaconda_client import (read_binstar_tokens,
                                            remove_binstar_token,
                                            set_binstar_token)
from conda.models.channel import Channel
from packaging import version

if sys.version_info[0] < 3:
    from urlparse import urljoin
else:
    from urllib.parse import urljoin

CONDA_VERSION = version.parse(conda.__version__)

REPO_URL = os.getenv('CONDA_TOKEN_REPO_URL', 'https://repo.anaconda.cloud/repo/')
MAIN_CHANNEL = 'main'
ACTIVE_CHANNELS = ['r', 'msys2']
ARCHIVE_CHANNELS = ['free', 'mro-archive', 'pro']

user_rc_path = abspath(expanduser('~/.condarc'))
escaped_user_rc_path = user_rc_path.replace("%", "%%")
escaped_sys_rc_path = abspath(join(sys.prefix, '.condarc')).replace("%", "%%")


class CondaTokenError(RuntimeError):
    pass


class CondaVersionWarning(UserWarning):
    pass


def can_restore_free_channel():
    return CONDA_VERSION >= version.parse('4.7.0')


def get_ssl_verify():
    context = reset_context()
    return context.ssl_verify


def clean_index():
    """Runs conda clean -i.

    It is important to remove index cache when
    changing the condarc to ensure that the downloaded
    repodata is correct.
    """
    run_command(Commands.CLEAN, '-i')


def validate_token(token, no_ssl_verify=False):
    """Checks that token can be used with the repository."""

    # Read ssl_verify from condarc
    ssl_verify = get_ssl_verify()

    # Force ssl_verify: false
    if no_ssl_verify:
        ssl_verify = False

    channel = Channel(urljoin(REPO_URL, 'main/noarch/repodata.json'))
    channel.token = token
    token_url = channel.url(with_credentials=True)

    r = requests.head(token_url, verify=ssl_verify)
    if r.status_code != 200:
        raise CondaTokenError('The token could not be validated. Please check that you have typed it correctly.')


def enable_extra_safety_checks(condarc_system=False, condarc_env=False, condarc_file=None):
    """Enable package signature verification.

    This will set extra_safety_checks: True and
    signing_metadata_url_base in the CondaRC file.
    """
    if CONDA_VERSION < version.parse('4.10.1'):
        warnings.warn('You need upgrade to at least Conda version 4.10.1 to enable package signature verification.',
                       CondaVersionWarning)
        return

    condarc_file_args = []
    if condarc_system:
        condarc_file_args.append('--system')
    elif condarc_env:
        condarc_file_args.append('--env')
    elif condarc_file:
        condarc_file_args.append('--file={}'.format(condarc_file))

    safety_check_args = ['--set', 'extra_safety_checks', 'true']
    safety_check_args.extend(condarc_file_args)
    run_command(Commands.CONFIG, *safety_check_args)

    metadata_url_args = ['--set', 'signing_metadata_url_base', REPO_URL.rstrip('/')]
    metadata_url_args.extend(condarc_file_args)
    run_command(Commands.CONFIG, *metadata_url_args)


def disable_extra_safety_checks(condarc_system=False, condarc_env=False, condarc_file=None):
    """Disable package signature verification.

    This will set extra_safety_checks: false and remove
    signing_metadata_url_base in the CondaRC file.
    """

    if CONDA_VERSION < version.parse('4.10.1'):
        return

    condarc_file_args = []
    if condarc_system:
        condarc_file_args.append('--system')
    elif condarc_env:
        condarc_file_args.append('--env')
    elif condarc_file:
        condarc_file_args.append('--file={}'.format(condarc_file))

    safety_check_args = ['--set', 'extra_safety_checks', 'false']
    safety_check_args.extend(condarc_file_args)
    try:
        run_command(Commands.CONFIG, *safety_check_args)
    except CondaKeyError:
        pass

    metadata_url_args = ['--remove-key', 'signing_metadata_url_base']
    metadata_url_args.extend(condarc_file_args)
    try:
        run_command(Commands.CONFIG, *metadata_url_args)
    except CondaKeyError:
        pass


def _set_add_anaconda_token(condarc_system=False,
                            condarc_env=False,
                            condarc_file=None):
    """Run conda config --set add_anaconda_token true.

    Setting this parameter to true ensures that the token
    is used when making requests to the repository.
    """
    config_args = ['--set', 'add_anaconda_token', 'true']

    if condarc_system:
        config_args.append('--system')
    elif condarc_env:
        config_args.append('--env')
    elif condarc_file:
        config_args.append('--file={}'.format(condarc_file))

    run_command(Commands.CONFIG, *config_args)


def _set_ssl_verify_false(condarc_system=False,
                          condarc_env=False,
                          condarc_file=None):
    """Run conda config --set ssl_verify false.

    Setting this parameter to false disables all
    SSL verification for conda activities
    """
    config_args = ['--set', 'ssl_verify', 'false']

    if condarc_system:
        config_args.append('--system')
    elif condarc_env:
        config_args.append('--env')
    elif condarc_file:
        config_args.append('--file={}'.format(condarc_file))

    run_command(Commands.CONFIG, *config_args)


def _unset_restore_free_channel(condarc_system=False,
                                condarc_env=False,
                                condarc_file=None):
    """Runs conda config --set restore_free_channel false.

    The free channel is provided by Commercial Edition as
    and should be added directly."""
    config_args = ['--set', 'restore_free_channel', 'false']

    if condarc_system:
        config_args.append('--system')
    elif condarc_env:
        config_args.append('--env')
    elif condarc_file:
        config_args.append('--file={}'.format(condarc_file))

    run_command(Commands.CONFIG, *config_args, use_exception_handler=True)


def _set_channel(channel, prepend=True,
                 condarc_system=False,
                 condarc_env=False,
                 condarc_file=None):
    """Adds a named Commercial Edition channel to default_channels."""
    channel_url = urljoin(REPO_URL, channel)

    config_args = [
        '--prepend' if prepend else '--append',
        'default_channels',
        channel_url
    ]

    if condarc_system:
        config_args.append('--system')
    elif condarc_env:
        config_args.append('--env')
    elif condarc_file:
        config_args.append('--file={}'.format(condarc_file))

    run_command(Commands.CONFIG, *config_args)


def _remove_default_channels(condarc_system=False,
                             condarc_env=False,
                             condarc_file=None):
    """Runs conda config --remove-key default_channels

    It is best to remove the default_channels in case they
    are not set at the default values before configuring
    Commercial Edition.
    """
    config_args = ['--remove-key', 'default_channels']

    if condarc_system:
        config_args.append('--system')
    elif condarc_env:
        config_args.append('--env')
    elif condarc_file:
        config_args.append('--file={}'.format(condarc_file))

    try:
        run_command(Commands.CONFIG, *config_args)
    except CondaKeyError:
        pass


def configure_default_channels(condarc_system=False,
                               condarc_env=False,
                               condarc_file=None,
                               include_archive_channels=None):
    """Configure the default_channels to utilize only Commercial Edition.


    This function performs the following actions
    1. unset default_channels if it exists in the condarc
    2. unset restore_free_channel if it exists in the condarc
    3. Add the main, r, and msys2 channels to default_channels
    4. Optionally add any of the archive channels:
       free, pro, mro, mro-archive
    """
    _remove_default_channels(condarc_system, condarc_env, condarc_file)

    if can_restore_free_channel():
        _unset_restore_free_channel(condarc_system, condarc_env, condarc_file)

    _set_channel(MAIN_CHANNEL, prepend=True,
                 condarc_system=condarc_system, condarc_env=condarc_env, condarc_file=condarc_file)

    for c in ACTIVE_CHANNELS:
        _set_channel(c, prepend=False,
                     condarc_system=condarc_system, condarc_env=condarc_env, condarc_file=condarc_file)

    if include_archive_channels is None:
        return

    for c in include_archive_channels:
        if c in ARCHIVE_CHANNELS:
            _set_channel(c, prepend=False,
                         condarc_system=condarc_system, condarc_env=condarc_env, condarc_file=condarc_file)
        else:
            raise ValueError('The archive channel %s is not one of %s' % (c, ', '.join(ARCHIVE_CHANNELS)))


def token_list():
    """Return a dictionary of tokens for all configured repository urls.

    Note that this function will return tokens configured for non-Commercial Edition
    urls."""
    return read_binstar_tokens()


def token_remove(system=False,
                 env=False,
                 file=None):
    """Completely remove the Commercial Edition token and default_channels.

    This function performs three actions.
    1. Remove the token
    2. Remove the custom default_channels in the condarc
    3. Disable package signature verification
    4. Run conda clean -i
    """
    remove_binstar_token(REPO_URL)
    _remove_default_channels(system, env, file)
    disable_extra_safety_checks(system, env, file)
    clean_index()


def token_set(token,
              system=False,
              env=False,
              file=None,
              include_archive_channels=None,
              no_ssl_verify=False,
              enable_signature_verification=False):
    """Set the Commercial Edition token and configure default_channels.


    This function performs 4 actions.
    1. Remove previous Commercial Edition token if present.
    2. Add token.
    3. Configure default_channels in the condarc file.
    4. Optionally enable Conda package signature verification
    5. Run conda clean -i
    """
    remove_binstar_token(REPO_URL)

    set_binstar_token(REPO_URL, token)
    _set_add_anaconda_token(system, env, file)

    if no_ssl_verify:
        _set_ssl_verify_false(system, env, file)

    if enable_signature_verification:
        enable_extra_safety_checks(system, env, file)

    configure_default_channels(system, env, file, include_archive_channels)
    clean_index()
