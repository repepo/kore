"""
Copy an object within channels in your Anaconda Repository.

example::

    conda repo copy <CHANNEL_NAME>::<PACKAGE_NAME>/<PACKAGE_VERSION>/<FILE_NAME> -d <ANOTHER_CHANNEL_NAME>/<SUBCHANNEL>

"""
from ..utils.artifacts import PackageSpec
from .bulk_action import BulkActionCommand


class SubCommand(BulkActionCommand):
    name = "copy"

    def add_parser(self, subparsers):
        parser = subparsers.add_parser(
            "copy",
            help="Copy packages from one channel to another",
            description=__doc__,
        )

        parser.add_argument(
            "specs",
            help=(
                "Package - written as "
                "<channel>/<subchannel>[::<package>[/<version>[/<filename>]]]"
                "If filename is not given, copy all files in the version"
            ),
            type=PackageSpec.from_string,
            nargs="+",
        )
        parser.add_argument(
            "-d", "--destination", help="Channel to put all packages into", default=None
        )
        parser.add_argument(
            "--family",
            default="conda",
            help="artifact family (i.e.: conda, python, cran, anaconda_project, "
            "anaconda_env, nootebook)",
        )
        parser.set_defaults(main=self.main)
