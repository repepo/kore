import unittest
from unittest.mock import patch

from repo_cli.commands.channel import SubCommand


@patch("repo_cli.commands.base.SubCommandBase")
class Test(unittest.TestCase):

    subcommand = None

    def setup(self, mock_command_base):
        self.subcommand = SubCommand(mock_command_base)
        self.subcommand.args.create = None
        self.subcommand.args.remove = None
        self.subcommand.args.list = None
        self.subcommand.args.list_files = None
        self.subcommand.args.list_packages = None
        self.subcommand.args.remove = None
        self.subcommand.args.show = None
        self.subcommand.args.history = None
        self.subcommand.args.lock = None
        self.subcommand.args.soft_lock = None
        self.subcommand.args.unlock = None
        self.subcommand.args.freeze = None
        self.subcommand.args.unfreeze = None

    def test_lock_channel(self, mock_command_base):
        mock_command_base.api.update_channel.return_value = ""
        self.setup(mock_command_base)
        self.subcommand.args.lock = "test_channel"
        self.subcommand.main()
        mock_command_base.api.update_channel.assert_called_once()
        mock_command_base.api.update_channel.assert_called_with(
            "test_channel",
            privacy="private",
            success_message="Channel test_channel is now locked",
        )

    def test_unlock_channel(self, mock_command_base):
        mock_command_base.api.update_channel.return_value = ""
        self.setup(mock_command_base)
        self.subcommand.args.unlock = "test_channel"
        self.subcommand.main()
        mock_command_base.api.update_channel.assert_called_once()
        mock_command_base.api.update_channel.assert_called_with(
            "test_channel",
            privacy="public",
            success_message="Channel test_channel is now unlocked",
        )

    def test_soft_lock_channel(self, mock_command_base):
        mock_command_base.api.update_channel.return_value = ""
        self.setup(mock_command_base)
        self.subcommand.args.soft_lock = "test_channel"
        self.subcommand.main()
        mock_command_base.api.update_channel.assert_called_once()
        mock_command_base.api.update_channel.assert_called_with(
            "test_channel",
            privacy="authenticated",
            success_message="Channel test_channel is now soft-locked",
        )
