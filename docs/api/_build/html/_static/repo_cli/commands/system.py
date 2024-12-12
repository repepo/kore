"""
System commands, use to display system information

Usage:
    --cleanup-blobs
    will clean up all blobs

    --cleanup-blobs sha1,sha1
    Diagnose blobs with given sha1

    --diagnose-blobs --channel-names channel1,channel2 --file-name filename.json
    download a json file with information about blobs of the given channels on Anaconda Server


"""
import json
import logging

from .base import SubCommandBase

logger = logging.getLogger("repo_cli")


class SubCommand(SubCommandBase):
    name = "system"

    def main(self):
        user_info = self.api.get_current_user()
        if not "admin" in user_info["roles"]:
            logger.error(
                "You are not an admin. This command is only available to admins."
            )
            return

        if self.args.cleanup_blobs is not None and len(self.args.cleanup_blobs) == 0:
            self.delete_blobs()
        if self.args.cleanup_blobs is not None:
            self.cleanup_blobs(self.args.cleanup_blobs)
        if self.args.diagnose_blobs:
            if not self.args.channel_names:
                self.log.error("please specify channel names")
            else:
                self.diagnose_blobs(self.args.channel_names, self.args.file_name)

    def cleanup_blobs(self, raw_args):
        if len(raw_args) == 0:
            logger.info("No blobs to cleanup")
            return

        blobs = raw_args[0].split(",")
        ret_val = self.api.post_blob_cleanup(blobs)

        logger.info(
            f"bloc count: {ret_val['blob_count']} space reclaimed: {ret_val['space_reclaimed']}"
        )

    def delete_blobs(self):
        ret_val = self.api.delete_blobs()
        logger.info(
            f"bloc count: {ret_val['blob_count']} space reclaimed: {ret_val['space_reclaimed']}"
        )

    def diagnose_blobs(self, channel_names, args_file_name=None):
        channel_names = channel_names.split(",")

        resp = self.api.diagnose_blobs(channel_names)
        if not resp.ok:
            logger.error("Error while diagnosing blobs")
            return

        data = resp.json()
        if args_file_name is None:
            filename = resp.headers.get("Content-Disposition").split("filename=")[1]
        else:
            filename = args_file_name

        with open(filename, "w") as f:
            json.dump(data, f)

        logger.info(f"Diagnosis file saved to {filename}")

    def add_parser(self, subparsers):
        self.subparser = subparsers.add_parser(
            "system", help="Return information about system", description=__doc__
        )

        self.subparser.add_argument(
            "--cleanup-blobs",
            help="Cleanup blobs",
            type=str,
            action="store",
            nargs="*",
        )

        self.subparser.add_argument(
            "--diagnose-blobs",
            help="Diagnose blobs",
            action="store_true",
        )

        self.subparser.add_argument(
            "--channel-names",
            help="List of channel names",
            type=str,
        )

        self.subparser.add_argument(
            "--file-name",
            help="File name",
            default=None,
            type=str,
        )

        self.subparser.set_defaults(main=self.main)
