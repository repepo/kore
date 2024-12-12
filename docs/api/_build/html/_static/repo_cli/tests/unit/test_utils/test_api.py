from unittest.mock import patch

import pytest
import requests_mock

from repo_cli.errors import InvalidName
from repo_cli.utils.api import RepoApi


@patch("requests.post")
def test_api_login_sucess(mock_post):
    with requests_mock.Mocker() as mock:
        mock.register_uri(
            "POST",
            "http://www.testurl.com/auth/login",
            json={"token": "test_token"},
            status_code=200,
        )

        api = RepoApi("http://www.testurl.com")
        assert api.login("test", "test") == "test_token"


@patch("requests.post")
def test_api_login_unauthorized(mock_post):
    with requests_mock.Mocker() as mock:
        mock.register_uri(
            "POST",
            "http://www.testurl.com/auth/login",
            status_code=401,
        )

        api = RepoApi("http://www.testurl.com")

        try:
            api.login("test", "test") == "test_token"
        except Exception as RepoCLIError:
            assert True


def test_is_subchannel():
    api = RepoApi("http://www.testurl.com")
    assert api.is_subchannel("main_channel") is False
    assert api.is_subchannel("main_channel/subchannel") is True


def test_get_channel_url():
    api = RepoApi("http://www.testurl.com")
    assert api._get_channel_url("test") == "http://www.testurl.com/channels/test"
    assert (
        api._get_channel_url("test/test")
        == "http://www.testurl.com/channels/test/subchannels/test"
    )


def test_validate_channel_name():
    api = RepoApi("http://www.testurl.com")

    # checking for valid channel names
    assert api._validate_channel_name("test") is None
    assert api._validate_channel_name("test-1-2_3_4") is None

    # checking for not valid channel names
    with pytest.raises(InvalidName) as excinfo:
        api._validate_channel_name("test1234öäü")
    assert "Channel name contains invalid sequence" in str(excinfo.value)

    with pytest.raises(InvalidName) as excinfo:
        api._validate_channel_name("test.:#+")
    assert "Channel name contains invalid sequence" in str(excinfo.value)


def test_validate_channel_name_for_subchannel():
    api = RepoApi("http://www.testurl.com")

    # checking for valid subchannel names
    assert api._validate_channel_name("test/test") is None
    assert api._validate_channel_name("test-1-2_3_4/test-1-2_3_4") is None

    # checking for not valid subchannel structure
    with pytest.raises(InvalidName) as excinfo:
        api._validate_channel_name("test/test/test")
    assert (
        "Channel name test/test/test is not valid. It contains more than one '/'"
        in str(excinfo.value)
    )

    # checking for not valid subchannel names
    with pytest.raises(InvalidName) as excinfo:
        api._validate_channel_name("test#.:/test")
    assert "Channel name contains invalid sequence" in str(excinfo.value)

    with pytest.raises(InvalidName) as excinfo:
        api._validate_channel_name("test/testöäü")
    assert "Channel name contains invalid sequence" in str(excinfo.value)


def test_get_report():
    with requests_mock.Mocker() as mock:
        mock.post(
            "http://www.testurl.com/reports/artifact_downloads",
            json={"downloaded_items": {"test": "test"}},
            status_code=200,
        )

        api = RepoApi("http://www.testurl.com")
        assert api.get_report(
            "2023-01-01", "2020-01-02", "username", "channel1,channel2", "json"
        ) == {"test": "test"}


def test_get_report_error():
    with requests_mock.Mocker() as mock:
        mock.post(
            "http://www.testurl.com/reports/artifact_downloads",
            json={"test": "test"},
            status_code=400,
        )

        api = RepoApi("http://www.testurl.com")
        assert (
            api.get_report("2023-01-01", "2020-01-02", "username", "channel1,channel2")
            is False
        )
