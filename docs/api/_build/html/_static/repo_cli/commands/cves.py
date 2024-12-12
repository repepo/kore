"""
Manage your Anaconda repository channels.
"""

from __future__ import print_function, unicode_literals

import argparse

from ..utils.format import CVEFilesFormatter, CVEFormatter
from .base import SubCommandBase


class SubCommand(SubCommandBase):
    name = "cves"

    def main(self):
        self.log.info("")
        args = self.args
        if args.list:
            self.show_list(args.offset, args.limit)
        elif args.show:
            self.show(args.show)
        elif args.show_files:
            self.show_files(args.show_files, args.offset, args.limit)
        else:
            raise NotImplementedError()

    def show_list(self, offset=0, limit=50):
        data = self.api.get_cves(offset, limit)
        self.log.info(CVEFormatter.format_list(data["items"]))
        self.log.info("")

    def show(self, cve):
        data = self.api.get_cve(cve)
        self.log.info(CVEFormatter.format_detail(data))
        self.log.info("")

    def show_files(self, cve_id, offset=0, limit=50):
        data = self.api.get_cve_files(cve_id, offset, limit)
        self.log.info(
            "Showing %d packages, associated with %s, starting from %d from total %d:"
            % (limit, cve_id, offset, data["total_count"])
        )
        self.log.info("")
        self.log.info(CVEFilesFormatter.format_list(data["items"]))
        self.log.info("")

    def add_parser(self, subparsers):
        subparser = subparsers.add_parser(
            self.name,
            help="Access Anaconda Repository {}s".format(self.name),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=__doc__,
        )

        group = subparser.add_mutually_exclusive_group(required=True)

        group.add_argument(
            "--list",
            action="store_true",
            help="list all {}s for a user".format(self.name),
        )
        group.add_argument(
            "--show",
            metavar=self.name.upper(),
            help="Show details about {}".format(self.name),
        )
        group.add_argument(
            "--show-files",
            metavar=self.name.upper(),
            help="Show files for {} (Use limit/offset additionally)".format(self.name),
        )

        subparser.add_argument(
            "-o",
            "--offset",
            default=0,
            type=int,
            help="Offset when displaying the results",
        )
        subparser.add_argument(
            "-l",
            "--limit",
            default=50,
            type=int,
            help="Offset when displaying the results",
        )

        subparser.set_defaults(main=self.main)
