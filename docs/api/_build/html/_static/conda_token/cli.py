"""
CLI for conda-token.
"""

from __future__ import print_function

import os
import sys
from argparse import ArgumentParser

from . import __version__, repo_config


def token_list(args):
    """Default function for list subparser."""
    tokens = {k: v for k, v in repo_config.token_list().items() if k == repo_config.REPO_URL}
    if not tokens:
        print('No tokens have been configured for {}'.format(repo_config.REPO_URL), file=sys.stderr)
        return 1

    for url, token in tokens.items():
        print(url, token)

    return 0


def token_set(args):
    try:
        repo_config.validate_token(args.token, no_ssl_verify=args.no_ssl_verify)
    except repo_config.CondaTokenError as e:
        print(e, file=sys.stderr)
        return 1

    repo_config.token_set(args.token, args.system, args.env, args.file, args.include_archive_channels,
                          args.no_ssl_verify, args.enable_signature_verification)
    print('Success! Your token was validated and Conda has been configured.')
    return 0


def token_remove(args):
    repo_config.token_remove()
    return 0


def condarc_path_args(parser):
    """Add condarc path arguments."""
    config_file_location_group = parser.add_argument_group(
        'Config File Location Selection',
        "Without one of these flags, the user config file at '%s' is used." % repo_config.escaped_user_rc_path
    )
    location = config_file_location_group.add_mutually_exclusive_group()
    location.add_argument(
        "--system",
        action="store_true",
        help="Write to the system .condarc file at '%s'." % repo_config.escaped_sys_rc_path,
    )
    location.add_argument(
        "--env",
        action="store_true",
        help="Write to the active conda environment .condarc file (%s). "
             "If no environment is active, write to the user config file (%s)."
             "" % (
                 os.getenv('CONDA_PREFIX', "<no active environment>").replace("%", "%%"),
                 repo_config.escaped_user_rc_path,
             ),
    )
    location.add_argument(
        "--file",
        action="store",
        help="Write to the given file."
    )


def cli(argv=None):
    parser = ArgumentParser('conda-token', usage='conda token',
                            description='Configure token access for Anaconda Commercial Edition')

    parser.add_argument(
        '-V', '--version',
        action='version',
        help='Show the conda-token version number and exit.',
        version="conda-token %s" % __version__,
    )

    subparser = parser.add_subparsers(help='Token commands')

    subparser_list = subparser.add_parser('list', help='List token if configured.')
    subparser_list.set_defaults(func=token_list)

    subparser_remove = subparser.add_parser('remove', help='Remove token and revert default_channels.')
    condarc_path_args(subparser_remove)
    subparser_remove.set_defaults(func=token_remove)

    subparser_set = subparser.add_parser('set', help='Set your token and configure default_channels.')
    subparser_set.add_argument('token', help='Your token.')
    subparser_set.add_argument('--include-archive-channels', choices=repo_config.ARCHIVE_CHANNELS,
                                help='Add archived channels to default_channels. '
                                    '\nAvailable channels are mro, mro-archive, free, and pro.',
                                nargs='+', default=None, metavar='CHANNEL_NAME')
    subparser_set.add_argument('--no-ssl-verify', help='Force disabling SSL verification. '
                                                       'By default conda-token will respect your '
                                                       'ssl_verify flag in your Conda settings. '
                                                       'Using this flag will add ssl_verify: false '
                                                       'to your .condarc file. If not already set.',
                                action='store_true'
                               )
    subparser_set.add_argument('--enable-signature-verification', action='store_true',
                               help='Enable Conda package signature verification. Default is off.')
    condarc_path_args(subparser_set)
    subparser_set.set_defaults(func=token_set)

    if len(sys.argv) == 1:
        sys.argv.append('--help')

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    cli(sys.argv[1:])
