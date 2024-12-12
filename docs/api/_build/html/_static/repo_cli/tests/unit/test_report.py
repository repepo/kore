import os
from unittest.mock import MagicMock, mock_open, patch

from repo_cli.commands.report import SubCommand


@patch("repo_cli.commands.base.SubCommandBase")
def test_download_report_json(mock_command_base, tmp_path):
    mock_command_base.api.get_report.return_value = {"json": "data"}
    mock_command_base.api.list_channels.return_value = {
        "items": [{"name": "channel1"}, {"name": "channel2"}]
    }
    subcommand = SubCommand(mock_command_base)
    my_file = tmp_path / "test.json"
    subcommand.download_report(
        "2020-01-01",
        "2020-01-02",
        "json",
        "user1,user2",
        "channel1,channel2",
        filename=my_file,
    )
    assert my_file.read_text() == '{"json": "data"}'


@patch("repo_cli.commands.base.SubCommandBase")
def test_download_report_csv(mock_command_base, tmp_path):
    mock_command_base.api.get_report.return_value = "c,s,v"
    mock_command_base.api.list_channels.return_value = {
        "items": [{"name": "channel1"}, {"name": "channel2"}]
    }
    subcommand = SubCommand(mock_command_base)
    my_file = tmp_path / "test.csv"
    subcommand.download_report(
        "2020-01-01",
        "2020-01-02",
        "csv",
        "user1,user2",
        "channel1,channel2",
        filename=my_file,
    )
    assert my_file.read_text() == "c,s,v"
