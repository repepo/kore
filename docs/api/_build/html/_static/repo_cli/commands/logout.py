"""
Log out from conda repo
"""
import logging

from ..utils.config import remove_token
from .base import SubCommandBase

logger = logging.getLogger("repo_cli")


class SubCommand(SubCommandBase):
    name = "logout"

    def main(self):
        if self.access_token or self.args.force:
            # call remove token that will remove the token from the current selected site...
            remove_token(self.args)
            if self.args.force:
                self.log.info("Access tokens were removed")
            else:
                self.log.info("Logout successful")
        else:
            self.log.info("You are not logged in")

    def add_parser(self, subparsers):
        self.subparser = subparsers.add_parser(
            "logout", help="Log out from your Anaconda repository", description=__doc__
        )
        self.subparser.add_argument(
            "-f",
            "--force",
            dest="force",
            action="store_true",
            help="Remove relevant tokens event if not logged in. Default False",
        )

        self.subparser.set_defaults(main=self.main)
