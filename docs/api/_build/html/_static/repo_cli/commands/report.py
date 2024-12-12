"""
Download artifact download reports from anaconda server

conda repo report

This command will download a report of all artifact downloads from the anaconda server and save it in the current working directory.
If you want to chane the download directory, use the --filename option.
"""
import datetime
import json
import logging
import os
from pathlib import Path

from ..utils.validators import check_date_format
from .base import SubCommandBase

logger = logging.getLogger("repo_cli")


class SubCommand(SubCommandBase):
    name = "report"

    def main(self):
        self.download_report(
            self.args.date_from,
            self.args.date_to,
            self.args.file_type,
            self.args.user_names,
            self.args.channels,
            self.args.filename if self.args.filename else None,
        )

    def download_report(
        self, date_from, date_to, file_type, usernames, channels, filename=None
    ):
        if date_from > date_to:
            raise ValueError(
                f"Invalid date range: The starting date {date_from} must be earlier than the ending date {date_to}."
            )

        usernames = usernames.split(",") if usernames else None
        channels = channels.split(",") if channels else None

        self.validate_channels(channels)

        logger.info("Downloading report for %s to %s" % (date_from, date_to))
        report_data = self.api.get_report(
            date_from, date_to, usernames, channels, file_type
        )

        if not report_data:
            logger.error("No report available for the given dates")
            return

        filepath = Path(self.get_filename(date_from, date_to, file_type, filename))
        logger.info(f"Saving report to {filepath}")
        if file_type == "json":
            filepath.write_text(json.dumps(report_data), encoding="utf-8")
        elif file_type == "csv":
            filepath.write_text(report_data, encoding="utf-8")

    @staticmethod
    def get_filename(date_from, date_to, file_type, filename=None):
        if filename:
            file = Path(filename)
            if not file.is_file():
                return os.path.abspath(filename)
            else:
                logger.error(
                    f"File {filename} already exists. Please specify a different filename."
                )
                raise FileExistsError
        filename = f"artifact_download_report_{date_from}_{date_to}.{file_type}"
        return os.path.join(os.getcwd(), filename)

    def validate_channels(self, channels):
        if not channels:
            return True

        all_channels = [
            channel["name"]
            for channel in self.api.list_channels()["items"]
            if channel["name"]
        ]

        for channel in channels:
            if channel not in all_channels:
                raise ValueError(f"Channel {channel} does not exist")

    def add_parser(self, subparsers):
        self.subparser = subparsers.add_parser(
            "report", help="download reports", description=__doc__
        )

        self.subparser.add_argument(
            "--date_from",
            type=check_date_format,
            default=None,
            help="starting date YYYY-mm-dd for the report",
            required=True,
        )

        self.subparser.add_argument(
            "--date_to",
            type=check_date_format,
            default=datetime.datetime.now().strftime("%Y-%m-%d"),
            help="end date YYYY-mm-dd for the report",
        )

        self.subparser.add_argument(
            "--file-type",
            type=str,
            default="json",
            choices=["json", "csv"],
            help="format of the report",
        )

        self.subparser.add_argument(
            "--user_names",
            type=str,
            help="comma separated list of users",
        )

        self.subparser.add_argument(
            "--channels",
            type=str,
            help="comma separated list of channels",
        )

        self.subparser.add_argument(
            "--filename",
            type=str,
            help="filename to save the report. Can contain full path for report to be saved in a different directory",
        )

        self.subparser.set_defaults(main=self.main)
