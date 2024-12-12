# pylint: disable=missing-function-docstring

"""
Usage:
    anaconda download notebook
    anaconda download user/notebook
"""

from __future__ import unicode_literals

import argparse
import logging

from binstar_client import errors
from binstar_client.utils import get_server_api
from binstar_client.utils.config import PackageType
from binstar_client.utils.notebook import parse, has_environment
from binstar_client.utils.notebook.downloader import Downloader

logger = logging.getLogger('binstar.download')


def add_parser(subparsers):
    description = 'Download notebooks from your Anaconda repository'
    parser = subparsers.add_parser(
        'download',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        help=description,
        description=description,
        epilog=__doc__
    )

    parser.add_argument(
        'handle',
        help='user/notebook',
        action='store'
    )

    parser.add_argument(
        '-f', '--force',
        help='Overwrite',
        action='store_true'
    )

    parser.add_argument(
        '-o', '--output',
        help='Download as',
        default='.'
    )
    pkg_types = ', '.join(pkg_type.value for pkg_type in PackageType)
    parser.add_argument(
        '-t', '--package-type',
        help='Set the package type [{0}]. Defaults to downloading all package types available'.format(pkg_types),
        action='append',
    )
    parser.set_defaults(main=main)


def main(args):
    aserver_api = get_server_api(args.token, args.site)
    username, notebook = parse(args.handle)
    username = username or aserver_api.user()['login']
    downloader = Downloader(aserver_api, username, notebook)
    packages_types = list(map(PackageType, args.package_type) if args.package_type else PackageType)

    try:
        download_files = downloader.list_download_files(packages_types, output=args.output, force=args.force)
        for download_file, download_dist in download_files.items():
            downloader.download(download_dist)
            logger.info('%s has been downloaded as %s', args.handle, download_file)
            if has_environment(download_file):
                logger.info('%s has an environment embedded.', download_file)
                logger.info('Run:')
                logger.info('    conda env create %s', download_file)
                logger.info('To install the environment in your system')
    except (errors.DestinationPathExists, errors.NotFound, errors.BinstarError, OSError) as err:
        logger.info(err)
