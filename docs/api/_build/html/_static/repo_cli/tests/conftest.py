# -*- coding: utf-8 -*-
# Copyright (C) 2019 Anaconda, Inc
import logging
from pathlib import Path
from random import randint
from re import findall
from unittest.mock import Mock

import pytest
import responses
import yaml

from repo_cli import errors, main
from repo_cli.utils import config
from repo_cli.utils.api import RepoApi

from .integration.testcase import load_mocks

here = Path(__file__).parent
logging.basicConfig(level=logging.INFO)

log = logging.getLogger("repo_cli")
log.setLevel(logging.DEBUG)

default_config = config.get_config()
test_site = default_config.get("default_site")
test_url = default_config.get("url")

DEFAULT_PACKAGE_PATH = str(here / "data/numpy-1.15.4-py37hacdab7b_0.tar.bz2")

try:
    with open(str(here.joinpath("test_config.yaml")), "r") as fp:
        _auth_config = yaml.load(fp, yaml.Loader)
        default_username = _auth_config["user1"]["username"]
        default_pwd = _auth_config["user1"]["password"]

        username_2 = _auth_config["user2"]["username"]
        password_2 = _auth_config["user2"]["password"]
except FileNotFoundError:
    print(
        "Configuration file test_config.yaml not found. Cannot run integration tests without users "
        "information. Please create your configuration file in the following format:"
        "user1:"
        "  password: password"
        "  username: username1"
        "user2:"
        "  password: password"
        "  username: username2"
    )

TOKEN_USER_1 = None
TOKEN_USER_2 = None


@pytest.fixture
def default_user(auth_config):
    return auth_config["user1"]


@pytest.fixture
def default_username(auth_config):
    return auth_config["user1"]["username"]


@pytest.fixture
def auth_config():
    return _auth_config


@pytest.fixture
def token_user_1(default_username):
    global TOKEN_USER_1

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        if TOKEN_USER_1:
            return TOKEN_USER_1

        token = config.load_token(test_site)
        if not token:
            main.main(
                [
                    "login",
                    "--username=%s" % default_username,
                    "--password=%s" % default_pwd,
                ]
            )
            token = config.load_token(test_site)

        TOKEN_USER_1 = token
        return token


@pytest.fixture
def token_user_2():
    global TOKEN_USER_2

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        if TOKEN_USER_2:
            return TOKEN_USER_2

        args = Mock(site=test_site)
        main.main(["logout"])
        main.main(["login", "--username=%s" % username_2, "--password=%s" % password_2])
        token = config.load_token(test_site)
        args = Mock(site=test_site)
        config.remove_token(args)

        # now we need to store the  user 1 token back so it can be used as "default" user
        config.store_token(TOKEN_USER_1, args)
        TOKEN_USER_2 = token

        return token


@pytest.fixture
def new_channel(token_user_1):
    channel_name = "test_channel_1234"
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        main.main(["-t", token_user_1, "channel", "--create", channel_name])

        yield channel_name

        main.main(["-t", token_user_1, "channel", "--remove", channel_name])


@pytest.fixture
def new_channel_with_package(token_user_1, new_channel):
    main.main(["-t", token_user_1, "upload", "-c", new_channel, DEFAULT_PACKAGE_PATH])

    return new_channel


@pytest.fixture
def many_channels(token_user_1):
    channel_names = []

    for x in range(5):
        just_a_number = randint(0, 9999)
        channel_name = f"test_channel_0{just_a_number}"
        main.main(["-t", token_user_1, "channel", "--create", channel_name])
        channel_names.append(channel_name)

    yield channel_names

    for channel_name in channel_names:
        main.main(["-t", token_user_1, "channel", "--remove", channel_name])


@pytest.fixture
def new_subchannel(token_user_1, new_channel):
    subchannel_name = "test_channel_1234"
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        main.main(
            [
                "-t",
                token_user_1,
                "channel",
                "--create",
                new_channel + "/" + subchannel_name,
            ]
        )

        yield new_channel + "/" + subchannel_name

        main.main(
            [
                "-t",
                token_user_1,
                "channel",
                "--remove",
                new_channel + "/" + subchannel_name,
            ]
        )


@pytest.fixture
def new_subchannel_with_package(token_user_1, new_subchannel):
    main.main(
        ["-t", token_user_1, "upload", "-c", new_subchannel, DEFAULT_PACKAGE_PATH]
    )

    return new_subchannel


@pytest.fixture
def package_path():
    return DEFAULT_PACKAGE_PATH


@pytest.fixture
def package_filename():
    return DEFAULT_PACKAGE_PATH.split("/")[-1]


@pytest.fixture
def package_name():
    return DEFAULT_PACKAGE_PATH.split("/")[-1].split("-")[0]


@pytest.fixture
def package_version():
    return DEFAULT_PACKAGE_PATH.split("/")[-1].split("-")[1]


@pytest.fixture
def CVE_ids(caplog, token_user_1):
    main.main(["-t", token_user_1, "cves", "--list"])

    return findall(r"CVE-\d{4}-\d{4}", caplog.text)


# Trying to return a function to work around passing arguements to fixtures.
# It would be nice to be able to just say give me a (sub)channel with package(s)/file(s)
# TODO:figure out a way to pass the has_package param.
@pytest.fixture
def experimental_new_channel(token_user_1):
    def new_channel(has_package=False):
        just_a_number = randint(0, 9999)
        channel_name = "test_channel_0{}".format(just_a_number)
        main.main(["-t", token_user_1, "channel", "--create", channel_name])

        if has_package:
            main.main(
                ["-t", token_user_1, "upload", "-c", channel_name, DEFAULT_PACKAGE_PATH]
            )

        return channel_name

    channel_name = new_channel()
    yield channel_name

    main.main(["-t", token_user_1, "channel", "--remove", channel_name])


# patch methods to force the tokens into place when the CLI asks for additional login info
# todo: this causes token issues with tests that don't use it, see CBR-1594
def run_patched(self):
    default_username = _auth_config["user1"]["username"]
    default_pwd = _auth_config["user1"]["password"]

    try:
        token = get_token_patched(default_username, default_pwd, test_url)
        self.username = default_username
        self._access_token = token
        self.api._access_token = self._access_token
        self.api._jwt = token

    # If there is weirdness related to tokens disable this exception suppression.
    except AttributeError:
        pass

    try:
        try:
            if not hasattr(self.args, "main"):
                self.parser.error(
                    "A sub command must be given. "
                    "To show all available sub commands, run:\n\n\t conda repo -h\n"
                )

            return self.args.main()

        except errors.Unauthorized:

            self.log.info(
                "The action you are performing requires authentication, "
                "please sign in:"
            )
            self._access_token = self.auth_manager.login()
            return self.args.main(self)

    except errors.ShowHelp:
        self.args.sub_parser.print_help()
        if exit:
            raise SystemExit(1)
        else:
            return 1


def get_token_patched(default_username, default_pwd, test_url):
    token_api = RepoApi(test_url)

    return token_api.login(default_username, default_pwd)
