import json
import os

import responses

from repo_cli.utils import config

default_config = config.get_config()
url = default_config.get("url", "http://localhost:8088/api/")


REQUESTS_LIST = (
    (responses.POST, f"{url}auth/login", "login.json", 200),
    (responses.GET, f"{url}account/tokens", "tokens.json", 200),
    (responses.POST, f"{url}account/tokens", "tokens_put.json", 200),
    (
        responses.PUT,
        f"{url}account/tokens/37233609-9ca2-4fde-b75f-9c6b4de86d89",
        "tokens_put.json",
        200,
    ),
    (responses.GET, f"{url}account/scopes", "scopes.json", 200),
    (responses.POST, f"{url}channels", "create_channel.json", 201),
    (responses.DELETE, f"{url}channels/test_channel_1234", None, 202),
    (
        responses.POST,
        f"{url}channels/test_channel_1234/subchannels",
        "create_subchannel.json",
        201,
    ),
    (
        responses.DELETE,
        f"{url}channels/test_channel_1234/subchannels/test_channel_1234",
        None,
        202,
    ),
    (responses.GET, f"{url}account/channels", "channels.json", 200),
    (
        responses.POST,
        f"{url}channels/test_channel_1234/artifacts",
        "upload_artifact.json",
        201,
    ),
    (
        responses.GET,
        f"{url}channels/test_channel_1234",
        "get_channel_1234.json",
        200,
    ),
    (
        responses.GET,
        f"{url}channels/test_channel_1234/artifacts",
        "get_channel_artifacts.json",
        200,
    ),
    (
        responses.GET,
        f"{url}channels/test_channel_1234/artifacts/conda/numpy/files",
        "get_artifacts_conda_numpy.json",
        200,
    ),
    (
        responses.POST,
        f"{url}channels/test_channel_1234/test_channel_1234/artifacts",
        "upload_artifact_subchannel.json",
        201,
    ),
    (
        responses.GET,
        f"{url}channels/test_channel_1234/subchannels/test_channel_1234",
        "get_subchannel_1234.json",
        200,
    ),
    (
        responses.GET,
        f"{url}channels/test_channel_1234/subchannels/test_channel_1234/artifacts",
        "get_subchannel_artifacts.json",
        200,
    ),
    (
        responses.GET,
        f"{url}channels/test_channel_1234/subchannels/test_channel_1234/artifacts/conda/numpy/files",
        "get_subchannel_artifacts_conda_numpy.json",
        200,
    ),
    (
        responses.PUT,
        f"{url}channels/test_channel_1234",
        "lock_channel.json",
        200,
    ),
    (
        responses.PUT,
        f"{url}channels/test_channel_1234/subchannels/test_channel_1234",
        "lock_subchannel.json",
        200,
    ),
    (responses.GET, f"{url}account", "get_account.json", 200),
    (
        responses.POST,
        f"{url}channels/john/artifacts",
        "upload_channel_john.json",
        201,
    ),
    (
        responses.PUT,
        f"{url}channels/test_channel_1234/artifacts/bulk",
        None,
        202,
    ),
    (
        responses.PUT,
        f"{url}channels/test_channel_1234/subchannels/test_channel_1234/artifacts/bulk",
        None,
        202,
    ),
)


def get_fixture(filename: str):
    __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__))
    )
    f = os.path.join(__location__, "fixtures", filename)
    with open(f) as json_file:
        data = json.load(json_file)
        return data


def load_mocks(rsps):
    for req in REQUESTS_LIST:
        # mocker.register_uri(
        #     req[0],
        #     req[1],
        #     json=get_fixture(req[2]) if req[2] else "",
        #     status_code=req[3],
        # )

        rsps.add(
            req[0], req[1], json=get_fixture(req[2]) if req[2] else "", status=req[3]
        )
