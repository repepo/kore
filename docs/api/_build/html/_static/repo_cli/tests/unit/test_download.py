from unittest.mock import patch

import requests_mock

from repo_cli.commands.download import SubCommand


@patch("repo_cli.commands.base.SubCommandBase")
@patch("requests.post")
def test_download_notebook(mock_command_base, mock, tmp_path):
    mock_command_base.api.get_notebook_download_url.return_value = "http://testurl.com"
    subcommand = SubCommand(mock_command_base)
    my_file = tmp_path / "test.ipynb"

    with requests_mock.Mocker() as mock:
        mock.register_uri(
            "GET",
            "http://testurl.com",
            content="test".encode(),
            status_code=200,
        )
        subcommand.download_notebook("test_channel", "test_package", filename=my_file)

        assert my_file.read_text() == "test"
