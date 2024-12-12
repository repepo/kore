"""
It displays the username of the current user when this command is invoked
"""
import logging

import yaml

from .base import SubCommandBase

logger = logging.getLogger("repo_cli")


class SubCommand(SubCommandBase):
    name = "whoami"

    def main(self):
        if self.access_token:
            current_user = self.api.get_current_user()
            self.log.info(yaml.dump(current_user, default_flow_style=False))
        else:
            self.log.info("You are not logged in")

    def add_parser(self, subparsers):
        self.subparser = subparsers.add_parser(
            "whoami", help="Return information about logged user", description=__doc__
        )

        self.subparser.set_defaults(main=self.main)
