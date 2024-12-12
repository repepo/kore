"""

    conda repo upload CONDA_PACKAGE_1.bz2

##### See Also

  * [Uploading a Conda Package](https://team-docs.anaconda.com/en/latest/user/package.html#upload-pack)


"""
from __future__ import unicode_literals

import argparse
import logging
import os
from glob import glob
from os.path import basename, join

import requests

from .. import errors
from ..utils.config import PACKAGE_TYPES
from ..utils.detect import detect_package_type
from .base import SubCommandBase

logger = logging.getLogger("repo_cli")


def verbose_package_type(pkg_type, lowercase=True):
    verbose_type = PACKAGE_TYPES.get(pkg_type, "unknown")
    if lowercase:
        verbose_type = verbose_type.lower()
    return verbose_type


def determine_package_type(filename, args):
    """
    return the file type from the inspected package or from the
    -t/--package-type argument
    """
    if args.package_type:
        package_type = args.package_type
    else:
        logger.info("Detecting file type...")

        package_type = detect_package_type(filename)

        if package_type is None:
            message = (
                "Could not detect package type of file %r please specify package "
                "type with option --package-type" % filename
            )
            logger.error(message)
            raise errors.RepoCLIError(message)

        logger.info('File type is "%s"', package_type)

    return package_type


def get_default_channel(base_url, token, verify_ssl):
    url = join(base_url, "account", "me")
    logger.debug(f"[UPLOAD] Getting user default channel from {url}")
    response = requests.get(url, headers={"X-Auth": f"{token}"}, verify=verify_ssl)
    return response


def upload_file(base_url, token, filepath, channel, verify_ssl):
    url = join(base_url, "channels", channel, "artifacts")
    statinfo = os.stat(filepath)
    filename = basename(filepath)
    logger.debug(f"[UPLOAD] Using token {token} on {base_url}")
    multipart_form_data = {
        "content": (filename, open(filepath, "rb")),
        "filetype": (None, "conda1"),
        "size": (None, statinfo.st_size),
    }
    logger.info(f"Uploading to {url}...")
    response = requests.post(
        url,
        files=multipart_form_data,
        headers={"X-Auth": f"{token}"},
        verify=verify_ssl,
    )
    return response


def windows_glob(item):
    if os.name == "nt" and "*" in item:
        return glob(item)
    else:
        return [item]


class SubCommand(SubCommandBase):
    name = "upload"

    def main(self):
        # config = get_config(site=args.site)
        # url = config.get('url')
        if not self.access_token:
            raise errors.Unauthorized

        channels = self.args.channels
        if not channels:
            # In this case the user didn't specify any channel. Means we need to get
            # the user default channel
            channels = [self.api.get_default_channel()]
            if not channels[0]:
                raise errors.NoDefaultChannel(
                    "User default channel is not specified. Please set it in the user account or use -c option."
                )

        for filepath in self.args.files:
            for fp in filepath:
                for channel in channels:
                    logger.debug(f"Using token {self.access_token}")
                    package_type = determine_package_type(fp, self.args)
                    resp = self.api.upload_file(fp, channel, package_type)
                    if resp.status_code in [201, 200]:
                        logger.info(
                            f"File {fp} successfully uploaded to {self.api.base_url}/repo{channel} with response {resp.status_code}"
                        )
                        logger.debug(f"Server responded with {resp.content}")
                    else:
                        if resp.status_code == 401:
                            raise errors.Unauthorized()
                        else:
                            msg = (
                                f"Error uploading {fp} to {self.api.base_url}::{channel}. "
                                f"Server responded with status code {resp.status_code}.\n"
                                f"Error details: {resp.content}\n"
                            )
                            logger.error(msg)

    def add_parser(self, subparsers):
        description = "Upload packages to your Anaconda Server repository"
        self.subparser = parser = subparsers.add_parser(
            "upload",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            help=description,
            description=description,
            epilog=__doc__,
        )
        parser.add_argument(
            "files",
            nargs="+",
            help="Distributions to upload",
            default=[],
            type=windows_glob,
        )

        parser.add_argument(
            "-c",
            "--channel",
            action="append",
            default=[],
            metavar="CHANNELS",
            dest="channels",
        )
        parser.add_argument(
            "--no-progress", help="Don't show upload progress", action="store_true"
        )

        mgroup = parser.add_argument_group("metadata options")

        # To preserve current behavior
        pkgs = PACKAGE_TYPES.copy()
        # pkgs.pop('conda')
        # pkgs.pop('pypi')
        pkg_types = ", ".join(list(pkgs.keys()))
        mgroup.add_argument(
            "-t",
            "--package-type",
            help="Set the package type [{0}]. Defaults to autodetect".format(pkg_types),
        )

        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "-i",
            "--interactive",
            action="store_const",
            help="Run an interactive prompt if any packages are missing",
            dest="mode",
            const="interactive",
        )
        # group.add_argument('-f', '--fail', help='Fail if a package or release does not exist (default)',
        #                    action='store_const', dest='mode', const='fail')
        # group.add_argument('--force', help='Force a package upload regardless of errors',
        #                    action='store_const', dest='mode', const='force')
        # group.add_argument('--skip-existing', help='Skip errors on package batch upload if it already exists',
        #                    action='store_const', dest='mode', const='skip')

        parser.set_defaults(main=self.main)
