from . import commands
from ._version import get_versions
from .commands.base import RepoCommand


def main(args=None, exit=True):
    main_cmd = RepoCommand(commands, args, get_versions())
    main_cmd.run()


if __name__ == "__main__":
    main()
