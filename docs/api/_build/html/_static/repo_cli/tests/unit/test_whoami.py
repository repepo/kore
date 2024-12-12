import logging

import requests_mock

from repo_cli.commands.base import SubCommandBase
from repo_cli.commands.whoami import SubCommand
from repo_cli.utils.api import RepoApi


class TestWhoamiParent:
    url = "http://www.testurl.com/"
    api = None

    def __init__(self, token):
        self._access_token = token
        self.api = RepoApi(
            base_url=self.url, user_token=self._access_token, verify_ssl=False
        )
        self.api._access_token = self._access_token
        self.log = logging.getLogger(__name__)


def test_whoami_without_token(caplog):
    parent = TestWhoamiParent(None)
    whami = SubCommand(SubCommandBase(parent))
    whami.parent._access_token = False
    whami.main()

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    assert caplog.records[0].message == "You are not logged in"


def test_whoami_with_login(caplog):
    with requests_mock.Mocker() as mock:
        mock.get(
            "http://www.testurl.com/account",
            json={
                "user_id": "123456",
                "username": "some_name",
                "roles": ["admin", "default-roles-dev", "author"],
                "default_channel_name": "test",
                "dbroles": ["admin", "default-roles-dev"],
            },
            status_code=200,
            headers={
                "Server": "nginx/1.22.0",
                "Date": "Mon, 06 Feb 2023 14:46:53 GMT",
                "Content-Type": "application/json; charset=utf-8",
                "Content-Length": "292",
                "Connection": "keep-alive",
                "Cf-Team": "16c8cac7660000452e20d6a400000001",
            },
        )

        parent = TestWhoamiParent("some_token")
        whoami = SubCommand(SubCommandBase(parent))
        whoami.parent._access_token = True
        whoami.main()

        assert len(caplog.records) == 2

        record = caplog.records[0]
        assert record.levelno == logging.DEBUG
        assert (
            record.message
            == "[UPLOAD] Getting current user from http://www.testurl.com/account"
        )

        record = caplog.records[1]
        assert record.levelno == logging.INFO
        assert "dbroles:\n- admin\n- default-roles-dev" in record.message
        assert "default_channel_name: test" in record.message
        assert "roles:\n- admin\n- default-roles-dev\n- author" in record.message
        assert "user_id: '123456'" in record.message
        assert "username: some_name" in record.message
