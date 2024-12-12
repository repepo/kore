from pathlib import Path
from re import findall
from unittest.mock import patch

import pytest
import responses

from repo_cli import main

# Workaround for inability to call fixture within patch decorator
from repo_cli.tests.conftest import default_pwd, default_username, run_patched
from repo_cli.utils import config

from .testcase import load_mocks

here = Path(__file__).parent

default_config = config.get_config()
test_site = default_config.get("default_site", "test")

FILES_DIR = here.parent.joinpath("data")

site_config = config.get_config(site=test_site)
test_url = site_config.get("url", "http://localhost:8088/api/")

test_external_mirror_url = "https://conda.anaconda.org/conda-test"


@pytest.mark.skip(reason="CBR-5429")
def test_upload_channel_package(caplog, token_user_1, new_channel, package_path):
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        main.main(["-t", token_user_1, "upload", "-c", new_channel, package_path])

        assert (
            f"Uploading conda artifact to {test_url}channels/{new_channel}/artifacts..."
            in caplog.text
        ), caplog.text
        assert (
            f"File {package_path} successfully uploaded to {test_url}/repo{new_channel} with response 201"
            in caplog.text
        ), caplog.text


@pytest.mark.skip(reason="CBR-5429")
def test_upload_default_channel_package(
    caplog, token_user_1, package_path, default_username
):
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        main.main(["-t", token_user_1, "upload", package_path])

        print(
            "NEW HERE",
            f"Uploading conda artifact to {test_url}channels/{default_username}/artifacts...",
        )
        assert (
            f"Uploading conda artifact to {test_url}channels/{default_username}/artifacts..."
            in caplog.text
        ), caplog.text
        assert (
            f"File {package_path} successfully uploaded to {test_url}/repo{default_username} with response 201"
            in caplog.text
        ), caplog.text


# TODO: add upload to subchannel


@patch("builtins.input", lambda *args: "y")
@pytest.mark.skip(reason="CBR-5429")
def test_copy_channel_package(
    caplog,
    token_user_1,
    new_channel,
    new_channel_with_package,
    package_name,
    package_filename,
    package_version,
):

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        package_location = (
            new_channel_with_package
            + "::"
            + package_name
            + "/"
            + package_version
            + "/"
            + package_filename
        )

        main.main(["-t", token_user_1, "copy", package_location, "-d", new_channel])

        assert "copy action successful" in caplog.text, caplog.text


@patch("builtins.input", lambda *args: "y")
@pytest.mark.skip(reason="CBR-5429")
def test_copy_subchannel_package_(
    caplog,
    token_user_1,
    new_subchannel,
    new_subchannel_with_package,
    package_name,
    package_filename,
    package_version,
):

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        package_location = (
            new_subchannel_with_package
            + "::"
            + package_name
            + "/"
            + package_version
            + "/"
            + package_filename
        )

        main.main(["-t", token_user_1, "copy", package_location, "-d", new_subchannel])

        assert "copy action successful" in caplog.text, caplog.text


@patch("builtins.input", lambda *args: "y", default_username, default_pwd)
@pytest.mark.skip(reason="CBR-5429")
def test_move_channel_package(
    caplog,
    token_user_1,
    new_channel,
    new_channel_with_package,
    package_name,
    package_filename,
    package_version,
):

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        package_location = (
            new_channel_with_package
            + "::"
            + package_name
            + "/"
            + package_version
            + "/"
            + package_filename
        )

        main.main(["-t", token_user_1, "move", package_location, "-d", new_channel])

        assert "move action successful" in caplog.text, caplog.text


@patch("builtins.input", lambda *args: "y", default_username, default_pwd)
@pytest.mark.skip(reason="CBR-5429")
def test_move_subchannel_package(
    caplog,
    token_user_1,
    new_subchannel,
    new_subchannel_with_package,
    package_name,
    package_filename,
    package_version,
):

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
    package_location = (
        new_subchannel_with_package
        + "::"
        + package_name
        + "/"
        + package_version
        + "/"
        + package_filename
    )

    main.main(["-t", token_user_1, "move", package_location, "-d", new_subchannel])

    assert "move action successful" in caplog.text, caplog.text


@pytest.mark.skip(reason="CBR-5429")
def test_view_channel_package_details(
    caplog,
    token_user_1,
    new_channel_with_package,
    package_filename,
    package_name,
    package_version,
):

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        main.main(
            [
                "-t",
                token_user_1,
                "channel",
                "--list-file",
                new_channel_with_package + "::" + package_name,
                "--full-details",
            ]
        )

        assert (
            f"Found 1 files matching the specified spec {new_channel_with_package}:"
            in caplog.text
        ), caplog.text
        assert f"'{package_filename}'" in caplog.text.strip(), caplog.text.strip()
        assert f"'{package_name}'" in caplog.text, caplog.text
        assert f"'{package_version}'" in caplog.text, caplog.text


@pytest.mark.skip(reason="CBR-5429")
def test_view_subchannel_package_details(
    caplog,
    token_user_1,
    new_subchannel_with_package,
    package_filename,
    package_name,
    package_version,
):
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        main.main(
            [
                "-t",
                token_user_1,
                "channel",
                "--list-file",
                new_subchannel_with_package + "::" + package_name,
                "--full-details",
            ]
        )

        assert (
            f"Found 1 files matching the specified spec {new_subchannel_with_package}:"
            in caplog.text
        ), caplog.text
        assert f"'fn': '{package_filename}'" in caplog.text, caplog.text
        assert f"'name': '{package_name}'"
        assert f"'version': '{package_version}'"


@patch("builtins.input", lambda *args: "y")
@pytest.mark.skip(reason="CBR-5429")
def test_delete_channel_package(
    caplog,
    token_user_1,
    new_channel_with_package,
    package_filename,
    package_name,
    package_version,
):
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        package_location = f"{new_channel_with_package}::{package_name}/{package_version}/{package_filename}"
        main.main(["-t", token_user_1, "remove", package_location])

        assert (
            f"Spec {package_location} succesfully removed" in caplog.text
        ), caplog.text


@patch("builtins.input", lambda *args: "y")
@pytest.mark.skip(reason="CBR-5429")
def test_delete_subchannel_package(
    caplog,
    token_user_1,
    new_subchannel_with_package,
    package_filename,
    package_name,
    package_version,
):
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        package_location = f"{new_subchannel_with_package}::{package_name}/{package_version}/{package_filename}"
        main.main(["-t", token_user_1, "remove", package_location])

        assert (
            f"Spec {package_location} succesfully removed" in caplog.text
        ), caplog.text


@pytest.mark.skip(reason="CBR-1544")
def test_active_mirror_channel_from_internal(
    caplog, new_channel, new_channel_with_package
):
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        destination_channel = new_channel
        source = new_channel_with_package
        main.main(
            [
                "mirror",
                "--create",
                "test_mirror",
                "-c",
                destination_channel,
                "-s",
                source,
                "--mode",
                "active",
            ]
        )

        assert (
            f"Mirror test_mirror successfully created on channel {destination_channel}"
            in caplog.text
        ), caplog.text


@pytest.mark.skip(reason="CBR-1544")
def test_passive_mirror_channel_from_internal(
    caplog, new_channel, new_channel_with_package
):
    destination_channel = new_channel
    source_channel = new_channel_with_package
    main.main(
        [
            "mirror",
            "--create",
            "test_mirror_name",
            "-c",
            destination_channel,
            "-s",
            source_channel,
            "--mode",
            "passive",
        ]
    )

    assert f"Mirror test_mirror successfully created on channel {destination_channel}"


@pytest.mark.skip(reason="CBR-1594")
def test_active_mirror_channel_from_external(caplog, monkeypatch, new_channel):
    monkeypatch.setattr(main.RepoCommand, "run", run_patched)

    destination_channel = new_channel
    source = test_external_mirror_url
    main.main(
        [
            "mirror",
            "--create",
            "test_mirror",
            "-c",
            destination_channel,
            "-s",
            source,
            "--mode",
            "active",
        ]
    )

    assert (
        f"Mirror test_mirror successfully created on channel {destination_channel}"
        in caplog.text
    ), caplog.text


@pytest.mark.skip(reason="CBR-1594")
def test_passive_mirror_channel_from_external(caplog, monkeypatch, new_channel):
    monkeypatch.setattr(main.RepoCommand, "run", run_patched)

    destination_channel = new_channel
    source = test_external_mirror_url
    main.main(
        [
            "mirror",
            "--create",
            "test_mirror",
            "-c",
            destination_channel,
            "-s",
            source,
            "--mode",
            "passive",
        ]
    )

    assert (
        f"Mirror test_mirror successfully created on channel {destination_channel}"
        in caplog.text
    ), caplog.text


@pytest.mark.skip(reason="CBR-1544")
def test_delete_active_mirror_external(caplog, channel_with_active_external_mirror):

    main.main(["mirror", "--delete", channel_with_active_external_mirror])

    assert (
        f"Mirror test_mirror successfully deleted on channel {channel_with_active_external_mirror}"
        in caplog.text
    ), caplog.text


@pytest.mark.skip(reason="CBR-1544")
def test_delete_passive_mirror_external(caplog, channel_with_passive_external_mirror):

    main.main(["mirror", "--delete", channel_with_passive_external_mirror])

    assert (
        f"Mirror test_mirror successfully deleted on channel {channel_with_passive_external_mirror}"
        in caplog.text
    ), caplog.text


@pytest.mark.skip(reason="CBR-1544")
def test_delete_active_mirror_internal(caplog, channel_with_active_internal_mirror):

    main.main(["mirror", "--delete", channel_with_active_internal_mirror])

    assert (
        f"Mirror test_mirror successfully deleted on channel {channel_with_active_internal_mirror}"
        in caplog.text
    ), caplog.text


@pytest.mark.skip(reason="CBR-1544")
def test_delete_passive_channel_mirror_from_internal(
    caplog, channel_with_passive_internal_mirror
):

    main.main(["mirror", "--delete", channel_with_passive_internal_mirror])

    assert (
        f"Mirror test_mirror successfully deleted on channel {channel_with_passive_internal_mirror}"
        in caplog.text
    ), caplog.text


@pytest.mark.skip(reason="CBR-1544")
def test_delete_active_subchannel_mirror_external(
    caplog, subchannel_with_active_external_mirror
):

    main.main(["mirror", "--delete", subchannel_with_active_external_mirror])

    assert (
        f"Mirror test_mirror successfully deleted on channel {subchannel_with_active_external_mirror}"
        in caplog.text
    ), caplog.text


@pytest.mark.skip(reason="CBR-1544")
def test_delete_passive_subchannel_mirror_external(
    caplog, subchannel_with_passive_external_mirror
):

    main.main(["mirror", "--delete", subchannel_with_passive_external_mirror])

    assert (
        f"Mirror test_mirror successfully deleted on channel {subchannel_with_passive_external_mirror}"
        in caplog.text
    ), caplog.text


@pytest.mark.skip(reason="CBR-1544")
def test_delete_active_subchannel_mirror_internal(
    caplog, subchannel_with_active_internal_mirror
):

    main.main(["mirror", "--delete", subchannel_with_active_internal_mirror])

    assert (
        f"Mirror test_mirror successfully deleted on channel {subchannel_with_active_internal_mirror}"
        in caplog.text
    ), caplog.text


@pytest.mark.skip(reason="CBR-1544")
def test_delete_passive_subchannel_mirror_internal(
    caplog, subchannel_with_passive_internal_mirror
):

    main.main(["mirror", "--delete", subchannel_with_passive_internal_mirror])

    assert (
        f"Mirror test_mirror successfully deleted on channel {subchannel_with_passive_internal_mirror}"
        in caplog.text
    ), caplog.text


@pytest.mark.skip(reason="CBR-2029, cve service in intermediary state")
def test_list_CVEs(caplog, token_user_1):

    main.main(["-t", token_user_1, "cves", "--list"])

    for CVE in findall(r"CVE-\d{4}-\d{4}", caplog.text):
        assert CVE


@pytest.mark.skip(reason="CBR-2029, cve service in intermediary state")
def test_show_CVE(caplog, token_user_1, CVE_ids):

    CVE_id = CVE_ids[0]
    main.main(["-t", token_user_1, "cves", "--show", CVE_id])

    assert f"Cve: {CVE_id}" in caplog.text, caplog.text
