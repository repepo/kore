import logging
import os
from argparse import ArgumentParser
from pathlib import Path

import requests

from repo_cli.commands.base import SubCommandBase

logger = logging.getLogger("repo_cli")


class SubCommand(SubCommandBase):
    name = "download"

    def main(self):
        if self.args.channel and self.args.notebook:
            self.download_notebook(
                self.args.channel,
                self.args.notebook,
                self.args.filename if self.args.filename else None,
            )
        else:
            logger.info("Not enough parameters to identify download file")
            raise NotImplementedError()

    def download_notebook(
        self, channel: str, notebook_name: str, filename: os.path = None
    ):

        url = self.api.get_notebook_download_url(channel, notebook_name)
        r = requests.get(url, allow_redirects=True)

        open(self.get_filename(notebook_name, filename), "wb").write(r.content)

    @staticmethod
    def get_filename(notebook_name: str, filename: str = None):
        if filename:
            file = Path(filename)
            if not file.is_file():
                return os.path.abspath(filename)
            else:
                logger.error(
                    f"File {filename} already exists. Please specify a different filename."
                )
                raise FileExistsError
        filename = notebook_name + ".ipynb"
        return os.path.join(os.getcwd(), filename)

    def add_parser(self, subparsers: ArgumentParser):
        self.subparser = subparsers.add_parser(
            "download", help="Help string", description=__doc__
        )

        self.subparser.add_argument("--channel", type=str, help="Channel name")

        self.subparser.add_argument("--notebook", type=str, help="Notebook name")

        self.subparser.add_argument(
            "--filename",
            type=str,
            required=False,
            default=None,
            help="filename to save the report. Can contain full path for report to be saved in a different directory",
        )

        self.subparser.set_defaults(main=self.main)
