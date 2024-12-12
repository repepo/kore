"""
Download SBOM for artifact files
"""
import json
import logging

from .base import SubCommandBase

logger = logging.getLogger("repo_cli")


class SubCommand(SubCommandBase):
    name = "sbom"

    def main(self):
        if self.args.ckey:
            return self.download_sbom(self.args.channel, self.args.ckey)

        if (
            self.args.package
            and self.args.os
            and self.args.version
            and self.args.family
        ):
            return self.download_sbom_for_channel(
                self.args.channel,
                self.args.package,
                self.args.version,
                self.args.family,
                self.args.os,
            )

        logger.info("Not enough parameters to identify SBOM")
        logger.info("Either enter ckey (--ckey)")
        logger.info(
            "or specify package (--package), architecture (--os), version (--version) and familiy (--family)"
        )

    def download_sbom(self, channel, ckey: str):
        resp = self.api.get_sbom_download(channel, ckey)

        if resp.ok:
            data = resp.json()
            filename = resp.headers.get("Content-Disposition").split("filename=")[1]

            with open(filename, "w") as f:
                json.dump(data, f)

            logger.info(f"SBOM file saved to {filename}")

        if resp.status_code == 404:
            logger.info("No SBOM for " + str(ckey) + " available")

    def download_sbom_for_channel(self, channel, package, version, family, os):
        ckeys = self.get_ckey_for_package(channel, package, version, family, os)

        for ckey in ckeys:
            self.download_sbom(channel, ckey)

    def get_ckey_for_package(self, channel, package, version, family, os):
        # get the available ammount of packages in channel
        total_count_data = self.api.get_artifact_files(channel, package, family, 0)
        data = self.api.get_artifact_files(
            channel, package, family, total_count_data["total_count"]
        )

        packages = data.pop("items")
        ret = []

        for t in packages:
            if (
                t["metadata"]["repodata_record.json"]["version"] == version
                and t["metadata"]["repodata_record.json"]["subdir"] == os
            ):
                ret.append(t["ckey"].split("/")[1])

        return ret

    def add_parser(self, subparsers):
        self.subparser = subparser = subparsers.add_parser(
            "sbom", help="Get SBOM files", description=__doc__
        )

        subparser.add_argument(
            "--channel", required=True, help="set the channel containing the package"
        )

        subparser.add_argument("--package", help="set the package")

        subparser.add_argument(
            "--version", default="", help="set the version of the package"
        )

        subparser.add_argument(
            "--os", default="", help="set the architecture of the package"
        )

        subparser.add_argument(
            "--family",
            default="",
            help="Artifact family (i.e.: conda, python, cran, anaconda_project, anaconda_env, notebook).",
        )

        subparser.add_argument(
            "--ckey", default="", help="ckey to identify a artifact file"
        )

        subparser.set_defaults(main=self.main)
