from unittest.mock import patch

import pytest
import responses

from repo_cli import main
from repo_cli.tests.conftest import default_username, run_patched
from repo_cli.utils import config

from .testcase import load_mocks

default_config = config.get_config()
test_site = default_config.get("default_site", "test")

url = default_config.get("url", "http://localhost:8088/api/")


@pytest.mark.skip(reason="CBR-5429")
def test_login(default_user):
    main.main(["logout"])
    token = config.load_token(test_site)
    assert not token

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)

        main.main(
            [
                "login",
                "--username=%s" % default_user["username"],
                "--password=%s" % default_user["password"],
            ]
        )
        token = config.load_token(test_site)
        assert token


@pytest.mark.skip(reason="CBR-5429")
def test_logout(token_user_1):
    main.main(["logout"])
    token = config.load_token(test_site)
    assert not token


@pytest.mark.skip(reason="CBR-5429")
def test_list_auth_scopes(caplog):
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)

        main.main(["auth", "--list-scopes"])

        assert (
            "artifact:create,artifact:delete,artifact:download,artifact:edit,artifact:view,"
            "channel.default-channel:edit,channel.group:edit,channel:create,channel:delete,"
            "channel:edit,channel:history,channel:view,channel:view-artifacts,subchannel.group:edit,"
            "subchannel:create,subchannel:delete,subchannel:edit,subchannel:history,subchannel:view,"
            "subchannel:view-artifacts" in caplog.text
        ), caplog.text


@pytest.mark.skip(reason="CBR-5429")
def test_list_user_tokens(caplog, monkeypatch, token_user_1):
    monkeypatch.setattr(main.RepoCommand, "run", run_patched)

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)

        main.main(["auth", "--list"])

        assert "repo-cli-token" in caplog.text, caplog.text


@pytest.mark.skip(reason="CBR-5429")
def test_create_user_token(caplog, monkeypatch, token_user_1):
    monkeypatch.setattr(main.RepoCommand, "run", run_patched)
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        main.main(
            ["auth", "--create", "-n", "john", "-s", "artifact:view", "-s", "cve:view"]
        )
        assert (
            "Token 18ba3a4f7d8681adb5d0af712d14d77f2db5cccd6eb0cbc7 "
            "succesfully created with id: 6b39b019-8e28-4fa1-8249-7055cf36b9ef"
            in caplog.text
        ), caplog.text


@pytest.mark.skip("auth subcommand logger will cause failure")
@patch("builtins.input", lambda *args: default_username)
def test_remove_user_token(token_user_1):
    main.main(["-t", token_user_1, "auth", "--remove", token_user_1])
    token = config.load_token(test_site)
    assert not token
