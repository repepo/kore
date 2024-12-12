import sys

from repo_cli import errors
from repo_cli.commands.base import SubCommandBase
from repo_cli.utils import bool_input


class BulkActionCommand(SubCommandBase):
    name = None

    def main(self):

        for spec in self.args.specs:
            try:
                if spec._filename:
                    self.exec_bulk_action(
                        spec.channel,
                        self.args.family,
                        spec.package,
                        spec.version,
                        spec.filename,
                        spec,
                    )
                elif spec._version:
                    self.exec_bulk_action(
                        spec.channel,
                        self.args.family,
                        spec.package,
                        spec.version,
                        spec=spec,
                    )
                elif spec._package:
                    self.exec_bulk_action(
                        spec.channel, self.args.family, spec.package, spec=spec
                    )
                else:
                    self.log.error("Invalid package specification: %s", spec)

            except errors.NotFound:
                if self.args.force:
                    self.log.warning("", exc_info=True)
                    continue
                else:
                    raise

    def exec_bulk_action(
        self, channel, family, artifact, version=None, filename=None, spec=None
    ):
        base_item = {
            "name": artifact,
            "family": family,
        }

        target_description = ""
        if hasattr(self.args, "destination"):
            target_channel = self.args.destination
            if not target_channel:
                # destination channel not specified.. we need to get the user default channel and use it
                pass
            if target_channel:
                target_description = "to channel %s " % target_channel
        items = []
        if version or filename:
            packages, total_count = self.api.get_channel_artifacts_files(
                channel, family, artifact, version, filename
            )

            if not packages:
                self.log.warning(
                    "No files matches were found for the provided spec: %s\n" % (spec)
                )
                return

            files_descr = []
            for filep in packages:
                files_descr.append(
                    "PACKAGE: {name}:{version}-{ckey}; PLATFORM: {platform}; FILENAME: {fn}".format(
                        **filep
                    )
                )
                item = dict(base_item)
                item["ckey"] = filep["ckey"]
                items.append(item)

            affected_files = "\n".join(files_descr)

            msg = (
                "Are you sure you want to %s the package release %s %s? The following "
                "will be affected: \n\n %s\n\nConfirm?"
                % (self.name, target_description, spec, affected_files)
            )
        else:
            msg = "Conform action %s on spec %s ? (and all data with it?)" % (
                self.name,
                spec,
            )
            items = [base_item]
        force = getattr(self.args, "force", False)
        if force or bool_input(msg, False):
            self.api.channel_artifacts_bulk_actions(
                channel, self.name, items, target_channel=target_channel
            )
            self.log.info("%s action successful\n" % self.name)
        else:
            self.log.info("%s action not executed\n" % self.name)

    def add_parser(self, subparsers):
        raise NotImplementedError
