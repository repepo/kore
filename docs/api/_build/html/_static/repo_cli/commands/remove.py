"""
Remove an object from your Anaconda Server.

Example::


    # Remove specific package by providing version and package filename
    conda repo remove mychannel::mypackage/1.2.0/mypackage.tar.gz

    # Remove specific package by providing package filename
    conda repo remove mychannel::mypackage//mypackage.tar.gz

    # Remove all package files
    conda repo remove mychannel::mypackage

"""
import sys
from argparse import RawTextHelpFormatter

from .. import errors
from ..utils import bool_input
from ..utils.artifacts import PackageSpec
from .base import SubCommandBase

WAIT_SECONDS = 15


class SubCommand(SubCommandBase):
    name = "remove"

    def main(self):

        for spec in self.args.specs:
            try:
                if spec._filename:
                    self.remove_artifact(
                        spec.channel,
                        self.args.family,
                        spec.package,
                        spec.version,
                        spec.filename,
                        spec,
                    )
                elif spec._version:
                    self.remove_artifact(
                        spec.channel,
                        self.args.family,
                        spec.package,
                        spec.version,
                        spec=spec,
                    )
                elif spec._package:
                    self.remove_artifact(
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

    def remove_artifact(
        self, channel, family, artifact, version=None, filename=None, spec=None
    ):
        base_item = {
            "name": artifact,
            "family": family,
        }

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
                "Are you sure you want to remove the package release %s ? The following "
                "will be affected: \n\n %s\n\nConfirm?" % (spec, affected_files)
            )

        else:
            msg = (
                "Are you sure you want to remove the package %s ? (and all data with it?)"
                % (spec,)
            )
            items = [base_item]

        if self.args.force or bool_input(msg, False):
            self.api.channel_artifacts_bulk_actions(channel, "delete", items)

            self.log.info("Spec %s succesfully removed\n" % (spec))
        else:
            self.log.warning("Not removing release %s\n" % (spec))

    def add_parser(self, subparsers):

        parser = subparsers.add_parser(
            "remove",
            help="Remove an object from your Anaconda Server repository. Must refer to the "
            "formal package name as it appears in the URL of the package. Also "
            "use anaconda show <USERNAME> to see list of package names. "
            "Example: anaconda remove continuumio/empty-example-notebook",
            description=__doc__,
            formatter_class=RawTextHelpFormatter,
        )

        parser.add_argument(
            "specs",
            help="Package written as <channel>/<subchannel>[::<package>[/<version>[/<filename>]]]",
            type=PackageSpec.from_string,
            nargs="+",
        )
        parser.add_argument(
            "-f", "--force", help="Do not prompt removal", action="store_true"
        )
        parser.add_argument(
            "--family",
            default="conda",
            help="artifact family (i.e.: conda, python, cran, anaconda_project, "
            "anaconda_env, nootebook)",
        )

        parser.set_defaults(main=self.main)
