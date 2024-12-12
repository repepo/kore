import logging

import requests_mock

from repo_cli.commands.config import SubCommand


class TestConfigParent:
    def __init__(self):
        self.log = logging.getLogger(__name__)


def test_set_site_with_no_server(caplog):
    config = SubCommand(TestConfigParent())

    config.set_site({}, "test.com")

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.ERROR
    assert caplog.records[0].message == "URL is not valid"


def test_set_site_with_server(caplog):
    with requests_mock.Mocker() as mock:
        mock.get("https://www.testurl.com/", status_code=200)
        mock.get(
            "https://www.testurl.com/api/system",
            status_code=200,
            json={"service_name": "repo"},
        )
        config = SubCommand(TestConfigParent())

        config_file = {}
        config.set_site(config_file, "www.testurl.com")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.INFO
        assert (
            caplog.records[0].message
            == "Site https://www.testurl.com/api added as testurl.com to configuration"
        )


def test_set_site_with_server_with_invalid_response(caplog):
    with requests_mock.Mocker() as mock:
        mock.get("https://www.testurl.com/", status_code=200)
        mock.get(
            "https://www.testurl.com/api/system",
            status_code=200,
            json={"service_name": "some wrong response"},
        )
        config = SubCommand(TestConfigParent())

        config_file = {}
        config.set_site(config_file, "www.testurl.com")

        assert len(caplog.records) == 1
        assert caplog.records[0].levelno == logging.ERROR
        assert (
            caplog.records[0].message
            == "No Anaconda Server found at https://www.testurl.com/api"
        )


def test_extract_domain():
    config = SubCommand(TestConfigParent())

    assert config.extract_domain("https://www.testurl.com/api") == "testurl.com"
    assert config.extract_domain("https://www.testurl.com") == "testurl.com"
    assert config.extract_domain("https://www.testurl.com/") == "testurl.com"
    assert config.extract_domain("www.testurl.com") == "testurl.com"
    assert config.extract_domain("testurl.com") == "testurl.com"
    assert (
        config.extract_domain("https://www.something.testurl.com/api") == "testurl.com"
    )
