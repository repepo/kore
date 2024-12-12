from unittest.mock import patch

from repo_cli.commands.admin import SubCommand


@patch("repo_cli.commands.base.SubCommandBase")
def test_admin_set_conda_ident_basic(mock_command_base):
    mock_command_base.api.get_system_settings.return_value = {
        "anaconda_ident_settings": {}
    }

    subcommand = SubCommand(mock_command_base)
    subcommand.update_conda_ident_settings(True)

    mock_command_base.api.update_system_settings.assert_called_once()
    mock_command_base.api.update_system_settings.assert_called_with(
        {"anaconda_ident_settings": {"enabled": True}}
    )


@patch("repo_cli.commands.base.SubCommandBase")
def test_admin_set_conda_ident_with_existing_settings(mock_command_base):
    mock_command_base.api.get_system_settings.return_value = {
        "anaconda_ident_settings": {
            "enabled": False,
            "enable_from": "2020-01-01",
            "selected_tokens": ["a", "b", "c"],
        }
    }

    subcommand = SubCommand(mock_command_base)
    subcommand.update_conda_ident_settings(True)

    mock_command_base.api.update_system_settings.assert_called_once()
    mock_command_base.api.update_system_settings.assert_called_with(
        {
            "anaconda_ident_settings": {
                "enabled": True,
                "enable_from": "2020-01-01",
                "selected_tokens": ["a", "b", "c"],
            }
        }
    )


@patch("repo_cli.commands.base.SubCommandBase")
def test_admin_set_conda_ident_ste_all(mock_command_base):
    mock_command_base.api.get_system_settings.return_value = {
        "anaconda_ident_settings": {}
    }

    subcommand = SubCommand(mock_command_base)
    subcommand.update_conda_ident_settings(
        True, "2020-01-01", "client_token,session_token,environment_token,username"
    )

    mock_command_base.api.update_system_settings.assert_called_once()
    mock_command_base.api.update_system_settings.assert_called_with(
        {
            "anaconda_ident_settings": {
                "enabled": True,
                "enable_from": "2020-01-01",
                "selected_tokens": [
                    "client_token",
                    "session_token",
                    "environment_token",
                    "username",
                ],
            }
        }
    )
