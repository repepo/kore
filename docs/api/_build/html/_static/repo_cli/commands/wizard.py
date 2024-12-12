"""
Configure Conda to use Anaconda Server
"""

import argparse
import json
import os
import shutil
import subprocess
from os.path import abspath, exists, expanduser, join
from urllib.parse import urljoin

from .base import SubCommandBase


def root_prefix():
    output = subprocess.check_output("conda config --show --json", shell=True).decode()
    return json.loads(output).get("root_prefix", "")


class SubCommand(SubCommandBase):
    name = "wizard"
    user_rc_path = abspath(expanduser("~/.condarc"))
    escaped_user_rc_path = user_rc_path.replace("%", "%%")
    escaped_sys_rc_path = abspath(join(root_prefix(), ".condarc")).replace("%", "%%")
    escaped_env_rc_path = abspath(join(os.getenv("CONDA_PREFIX"), ".condarc")).replace(
        "%", "%%"
    )

    # flake8: noqa: C901
    def main(self):
        if self.args.system:
            self.condarc_args = "--system"
            self.condarc = self.escaped_sys_rc_path
        elif self.args.file:
            self.condarc_args = "--file %s" % self.args.file
            self.condarc = self.args.file
        elif self.args.env:
            self.condarc_args = "--env"
            self.condarc = self.escaped_env_rc_path
        else:
            self.condarc_args = ""
            self.condarc = self.escaped_user_rc_path

        self.condarc_backup = self.condarc + ".backup"

        if self.args.restore:
            self.restore_condarc()
            return

        self.log.info("Conda configuration wizard.")
        self.log.info(
            f"This wizard will configure your CondaRC file using "
            f'channels from {self.api.base_url.strip("/api")}'
        )
        self.log.info("")
        self.log.info(f"The CondaRC path is {self.condarc}")
        self.log.info("")

        all_channels = self.get_all_channels()

        self.log.info("The following channels are available:")
        mirror_channels = self.get_mirror_channels()
        non_mirror_channels = self.get_non_mirror_channels()
        print(f'{"Name":<30s} | Mirror | {"Privacy":15s} | Owners')
        print("-" * 75)
        for channel in mirror_channels:
            print(
                f'{channel["name"]:<30s} |  ︎ ✔    | {channel["privacy"]:15s} | {" ".join(channel["owners"])}'
            )

        for channel in non_mirror_channels:
            print(
                f'{channel["name"]:<30s} |  ︎      | {channel["privacy"]:15s} | {" ".join(channel["owners"])}'
            )

        default_channels = self.get_default_channels(all_channels)

        self.log.info("")
        self.log.info(
            'If you wish to add channels to the "channels" list\n'
            "provide a space-separated list. You may leave this blank."
        )
        value = input(": ")
        channels = []
        if value:
            channels = value.strip().split()

        self.log.info("The following Conda configuration will be applied")
        self.log.info(f'channel_alias: {urljoin(self.api.base_url, "/api/repo")}')
        self.log.info("default_channels:")
        for c in default_channels:
            self.log.info(f"  - {c}")
        self.log.info("channels:")
        self.log.info("  - defaults")
        if channels:
            for c in channels:
                self.log.info(f"  - {c}")

        self.log.info(f"Confirm changes to {self.condarc}")
        value = input("(The current condarc file will be archived) [Y, n]: ")
        if (value.lower() == "y") or (not value):
            self.backup_condarc()
            self.conda_config(
                "--set", "channel_alias", urljoin(self.api.base_url, "/api/repo")
            )

            self.conda_config("--remove-key", "default_channels")
            for c in default_channels:
                self.conda_config("--prepend", "default_channels", c)

            self.conda_config("--remove-key", "channels")
            self.conda_config("--prepend", "channels", "defaults")
            for c in channels:
                self.conda_config("--append", "channels", c)

    def backup_condarc(self):
        if exists(self.condarc):
            self.log.info(
                f"Backing up your current condarc file to {self.condarc_backup}"
            )
            shutil.copy(self.condarc, self.condarc_backup)

    def restore_condarc(self):
        if exists(self.condarc_backup):
            self.log.info(f"Restoring your condarc file from {self.condarc_backup}")
            shutil.copy(self.condarc_backup, self.condarc)
        else:
            self.log.info(f"No backup file {self.condarc_backup} was found.")
            self.log.info("There is nothing to do.")

    def conda_config(self, *args):
        cmd = ["conda", "config", self.condarc_args, *args]
        try:
            subprocess.check_output(" ".join(cmd), shell=True, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            if "CondaKeyError" in e.stderr.decode():
                pass

    def get_all_channels(self):
        return self.api.list_channels()["items"]

    def get_mirror_channels(self):
        channels = self.get_all_channels()
        return [c for c in channels if c["mirror_count"] > 0]

    def get_non_mirror_channels(self):
        channels = self.get_all_channels()
        return [c for c in channels if c["mirror_count"] == 0]

    def add_parser(self, subparsers):
        subparser = subparsers.add_parser(
            self.name,
            help="Configure Conda to use Anaconda Server",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=__doc__,
        )

        config_file_location_group = subparser.add_argument_group(
            "Config File Location Selection",
            (
                "Without one of these flags, the user config file at '%s' is used."
                % self.escaped_user_rc_path
            ),
        )
        location = config_file_location_group.add_mutually_exclusive_group()
        location.add_argument(
            "--system",
            action="store_true",
            help="Write to the system .condarc file at '%s'."
            % self.escaped_sys_rc_path,
        )
        location.add_argument(
            "--env",
            action="store_true",
            help="Write to the active conda environment .condarc file (%s). "
            "" % (self.escaped_env_rc_path),
        )
        location.add_argument("--file", action="store", help="Write to the given file.")
        subparser.add_argument(
            "--restore", help="Restore condarc from backup.", action="store_true"
        )

        subparser.set_defaults(main=self.main)

    def purge_channels(self, input_value, all_channels):
        channel_list = input_value.strip().split()
        all_channels_names = [c["name"] for c in all_channels]

        valid_channels = []
        invalid_channels = []
        for c in channel_list:
            if c in all_channels_names:
                valid_channels.append(c)
            else:
                invalid_channels.append(c)

        if len(invalid_channels) > 0:
            self.log.warning(
                "The following channels do not exist and will not be saved:"
            )
            for c in invalid_channels:
                self.log.warning(f"  - {c}")

        return valid_channels

    def get_default_channels(self, all_channels):
        while True:
            self.log.info("")
            self.log.info(
                "Provide a space-separated list of channels to set as default_channels\n"
                "It is common to add mirror channels here."
            )

            value = input(": ")
            default_channels = self.purge_channels(value, all_channels)

            if len(default_channels) > 0:
                break
            else:
                self.log.warning(
                    "At least one channel should be added to default channels! Try again..."
                )

        return default_channels
