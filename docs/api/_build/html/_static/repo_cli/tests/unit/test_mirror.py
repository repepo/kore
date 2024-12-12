from unittest.mock import patch

from requests import Response

from repo_cli.commands.mirror import SubCommand

mirror_resp = {
    "mirror_id": "8c6e75b3-48e5-427d-ab30-2c433ce89e30",
    "mirror_name": "test_mirror",
    "source_root": "https://conda.anaconda.org/waqasanjum",
    "mirror_type": "conda",
    "mirror_mode": "passive",
    "cron": "0 0 * * *",
    "proxy": None,
    "filters": {"only_signed": True},
    "state": "pending",
    "last_run_at": None,
    "created_at": "2023-04-11T15:08:26.646000+00:00",
    "updated_at": "2023-04-11T15:08:26.646000+00:00",
    "sync_state": {
        "logs": {
            "mirror_filtered": {
                "cve": [],
                "date": [],
                "signed": [],
                "license": [],
                "exclude_specs": [],
                "duplicated_legacy_conda": [],
            },
            "channel_filtered": {"license": [], "exclude_specs": []},
        },
        "count": {
            "active": 0,
            "failed": 0,
            "passive": 0,
            "removed": 0,
            "active_planned": 0,
            "mirror_filtered": {
                "cve": 0,
                "date": 0,
                "signed": 0,
                "license": 0,
                "only_specs": 0,
                "exclude_specs": 0,
                "duplicated_legacy_conda": 0,
            },
            "channel_filtered": {"license": 0, "exclude_specs": 0},
        },
        "last_ckey": None,
        "last_exception": None,
        "bytes_transfered": 0,
        "total_steps_count": 8,
        "current_step_number": 0,
        "current_step_percentage": 0,
        "current_step_description": "",
    },
}

mirror_test = {
    "name": "test_mirror",
    "id": "662ac922-4fb7-408a-b8bc-9a00125f14f9",
    "source_root": "https://conda.anaconda.org/jozi22",
    "type": "conda",
    "mode": "passive",
    "state": "running",
    "last_run_at": "2023-04-13T11:28:25.234745+00:00",
    "created_at": "2023-04-06T12:28:48.412000+00:00",
    "updated_at": "2023-04-13T11:28:25.980000+00:00",
    "sync_state": {
        "count": {
            "active": 0,
            "failed": 0,
            "passive": 0,
            "removed": 0,
            "active_planned": 0,
            "mirror_filtered": {
                "cve": 0,
                "date": 0,
                "signed": 0,
                "license": 0,
                "only_specs": 0,
                "exclude_specs": 0,
                "duplicated_legacy_conda": 0,
            },
            "channel_filtered": {"license": 0, "exclude_specs": 0},
        },
        "last_ckey": None,
        "last_exception": None,
        "bytes_transfered": 0,
        "total_steps_count": 8,
        "current_step_number": 3,
        "current_step_percentage": 100,
        "current_step_description": "Filter remote packages",
    },
    "cron": "0 0 1 * *",
    "channel": {"name": "isyed", "privacy": "public"},
}


@patch("repo_cli.commands.base.SubCommandBase")
def test_create_mirror_only_signed(mock_command_base):
    mock_command_base.api.create_mirror.return_value = mirror_resp

    subcommand = SubCommand(mock_command_base)
    subcommand.create_mirror(
        source="https://conda.anaconda.org/jozi22",
        channel="test_channel",
        name="test_create_mirror",
        run_now=True,
        type_="conda",
        mode="passive",
        proxy=None,
        cron=None,
        filter_licenses=None,
        filter_subdirs=None,
        filter_projects=None,
        filter_only_specs=None,
        filter_exclude_specs=None,
        filter_include_specs=None,
        cve_score=None,
        exclude_uncurrated_cve_packages=None,
        filter_only_signed=True,
        filter_date_from=None,
        filter_date_to=None,
    )
    mock_command_base.api.create_mirror.assert_called_once()
    mock_command_base.api.create_mirror.assert_called_with(
        "test_channel",
        "https://conda.anaconda.org/jozi22",
        "test_create_mirror",
        "passive",
        {"only_signed": True},
        "conda",
        None,
        True,
        None,
    )


@patch("repo_cli.commands.mirror.SubCommandBase")
@patch("repo_cli.commands.mirror.SubCommand._get_by_name")
def test_create_mirror_with_existing_mirror(mock_command, mock_command_base):
    mock_command.return_value = "mirror"

    subcommand = SubCommand(mock_command_base)
    subcommand.create_mirror(
        source="https://conda.anaconda.org/jozi22",
        channel="test_channel",
        name="test_create_mirror",
        run_now=True,
        type_="conda",
        mode="passive",
        proxy=None,
        cron=None,
        filter_licenses=None,
        filter_subdirs=None,
        filter_projects=None,
        filter_only_specs=None,
        filter_exclude_specs=None,
        filter_include_specs=None,
        cve_score=None,
        exclude_uncurrated_cve_packages=None,
        filter_only_signed=False,
        filter_date_from=None,
        filter_date_to=None,
    )
    mock_command.assert_called_once()
    mock_command_base.api.create_mirror.assert_not_called()


@patch("repo_cli.commands.base.SubCommandBase")
@patch("repo_cli.commands.mirror.SubCommand._get_by_name")
def test_update_mirror_only_signed(mock_command, mock_command_base):
    mock_command.return_value = {
        "mirror_id": "8c6e75b3-48e5-427d-ab30-2c433ce89e30",
        "mirror_name": "test_create_mirror",
        "source_root": "https://conda.anaconda.org/jozi22",
        "mirror_type": "conda",
        "mirror_mode": "passive",
        "cron": "0 0 * * *",
        "proxy": None,
        "filters": {"only_signed": False},
    }
    mock_command_base.api.update_mirror.return_value = mirror_resp

    subcommand = SubCommand(mock_command_base)
    subcommand.update_mirror(
        source="https://conda.anaconda.org/jozi22",
        channel="test_channel",
        name="test_create_mirror",
        new_name=None,
        run_now=True,
        type_="conda",
        mode="passive",
        proxy=None,
        cron=None,
        filter_licenses=None,
        filter_subdirs=None,
        filter_projects=None,
        filter_only_specs=None,
        filter_exclude_specs=None,
        filter_include_specs=None,
        cve_score=None,
        exclude_uncurrated_cve_packages=None,
        filter_only_signed=True,
        filter_date_from=None,
        filter_date_to=None,
    )
    mock_command_base.api.update_mirror.assert_called_once()
    mock_command_base.api.update_mirror.assert_called_with(
        "8c6e75b3-48e5-427d-ab30-2c433ce89e30",
        "test_channel",
        "https://conda.anaconda.org/jozi22",
        "test_create_mirror",
        "passive",
        {"only_signed": True},
        "conda",
        "0 0 * * *",
        True,
        None,
    )


@patch("repo_cli.commands.mirror.SubCommandBase")
@patch("repo_cli.commands.mirror.SubCommand._get_by_name")
def test_update_mirror_with_no_existing_mirror(mock_command, mock_command_base):
    mock_command.return_value = None

    subcommand = SubCommand(mock_command_base)
    subcommand.update_mirror(
        source="https://conda.anaconda.org/jozi22",
        channel="test_channel",
        name="test_create_mirror",
        new_name=None,
        run_now=True,
        type_="conda",
        mode="passive",
        proxy=None,
        cron=None,
        filter_licenses=None,
        filter_subdirs=None,
        filter_projects=None,
        filter_only_specs=None,
        filter_exclude_specs=None,
        filter_include_specs=None,
        cve_score=None,
        exclude_uncurrated_cve_packages=None,
        filter_only_signed=False,
        filter_date_from=None,
        filter_date_to=None,
    )
    mock_command.assert_called_once()
    mock_command_base.api.update_mirror.assert_not_called()


@patch("repo_cli.commands.base.SubCommandBase")
def test_stop_running_mirror(mock_command_base):
    mirror_data = {"items": [mirror_test]}
    mock_command_base.api.get_all_mirrors.return_value = mirror_data
    mock_command_base.api.stop_mirror.return_value = Response.ok

    subcommand = SubCommand(mock_command_base)
    mirror_data["items"][0]["state"] = "running"
    subcommand.stop(mirror_name="test_mirror")
    mock_command_base.api.get_all_mirrors.asser_called_once()
    mock_command_base.api.stop_mirror.assert_called_once()


@patch("repo_cli.commands.base.SubCommandBase")
def test_stop_pending_mirror(mock_command_base):
    mirror_data = {"items": [mirror_test]}
    mock_command_base.api.get_all_mirrors.return_value = mirror_data
    mock_command_base.api.stop_mirror.return_value = Response.ok

    subcommand = SubCommand(mock_command_base)

    mirror_data["items"][0]["state"] = "pending"
    subcommand.stop(mirror_name="test_mirror")
    mock_command_base.api.get_all_mirrors.asser_called_once()
    mock_command_base.api.stop_mirror.assert_called_once()


@patch("repo_cli.commands.base.SubCommandBase")
def test_stop_completed_mirror(mock_command_base):
    mirror_data = {"items": [mirror_test]}
    mirror_data["items"][0]["state"] = "completed"
    mock_command_base.api.get_all_mirrors.return_value = mirror_data
    mock_command_base.api.stop_mirror.return_value = Response.ok

    subcommand = SubCommand(mock_command_base)
    subcommand.stop(mirror_name="test_mirror")
    mock_command_base.api.get_all_mirrors.asser_called_once()
    mock_command_base.api.stop_mirror.assert_not_called()


@patch("repo_cli.commands.base.SubCommandBase")
def test_stop_stopped_mirror(mock_command_base):
    mirror_data = {"items": [mirror_test]}
    mirror_data["items"][0]["state"] = "stopped"
    mock_command_base.api.get_all_mirrors.return_value = mirror_data
    mock_command_base.api.stop_mirror.return_value = Response.ok

    subcommand = SubCommand(mock_command_base)
    subcommand.stop(mirror_name="test_mirror")
    mock_command_base.api.get_all_mirrors.asser_called_once()
    mock_command_base.api.stop_mirror.assert_not_called()


@patch("repo_cli.commands.base.SubCommandBase")
def test_restart_mirror(mock_command_base):
    mirror_data = {"items": [mirror_test]}
    mock_command_base.api.get_all_mirrors.return_value = mirror_data
    mock_command_base.api.update_mirror.return_value = mirror_test

    subcommand = SubCommand(mock_command_base)
    subcommand.restart(name="test_mirror")
    mock_command_base.api.get_all_mirrors.asser_called_once()
    mock_command_base.api.update_mirror.assert_called_once()


@patch("repo_cli.commands.base.SubCommandBase")
def test_restart_not_existing_mirror(mock_command_base):
    mirror_data = {"items": [mirror_test]}
    mock_command_base.api.get_all_mirrors.return_value = mirror_data
    mock_command_base.api.update_mirror.return_value = mirror_test

    subcommand = SubCommand(mock_command_base)
    subcommand.restart(name="test_mirror_not")
    mock_command_base.api.get_all_mirrors.asser_called_once()
    mock_command_base.api.update_mirror.assert_not_called()
