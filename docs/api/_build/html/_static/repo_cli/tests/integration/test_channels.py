import logging

import pytest
import responses

from repo_cli import main

# TODO: Parameterize tests for channel/subchannel.  Consider parameterizing fixtures as well.
from repo_cli.errors import InvalidName

from .testcase import load_mocks


@pytest.mark.skip(reason="CBR-5429")
def test_create_channel(caplog, token_user_1):
    channel = "test_channel_1234"
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        main.main(["-t", token_user_1, "channel", "--create", channel])

        # assert the logs of channel being successfully create are being called
        assert f"Channel {channel} successfully created" in caplog.text, caplog.text

        # This is to clean up after this test.
        # TODO:I couldn't find a way to accomplish this with a fixture as you cannot pass the channel name to the fixture as an arg.
        main.main(["-t", token_user_1, "channel", "--remove", channel])


@pytest.mark.skip(reason="CBR-5429")
def test_create_channel_with_invalid_character(token_user_1):
    with pytest.raises(InvalidName):
        channel = "pamela_anderson_&*_bobo"
        main.main(["-t", token_user_1, "channel", "--create", channel])


@pytest.mark.skip(reason="CBR-5429")
def test_delete_channel(caplog, token_user_1, new_channel):
    caplog.set_level(logging.INFO)
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        main.main(["-t", token_user_1, "channel", "--remove", new_channel])

        assert f"Channel {new_channel} successfully removed" in caplog.text, caplog.text


@pytest.mark.skip(reason="CBR-5429")
def test_create_subchannel(caplog, token_user_1, new_channel):
    subchannel_name = "test_channel_1475"

    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        main.main(
            [
                "-t",
                token_user_1,
                "channel",
                "--create",
                new_channel + f"/{subchannel_name}",
            ]
        )

        assert (
            f"Channel {new_channel}/{subchannel_name} successfully created"
            in caplog.text
        ), caplog.text


@pytest.mark.skip(reason="CBR-5429")
def test_delete_subchannel(caplog, token_user_1, new_subchannel):
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        main.main(["-t", token_user_1, "channel", "--remove", new_subchannel])

        assert (
            f"Channel {new_subchannel} successfully removed" in caplog.text
        ), caplog.text


# todo: improve this test with a unique test name creation, or return to the many_channels implementation after CBR-1701
@pytest.mark.skip(reason="CBR-5429")
def test_list_channels(caplog, token_user_1):
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        load_mocks(rsps)
        main.main(["-t", token_user_1, "channel", "--list"])

        assert "john" in caplog.text, caplog.text


@pytest.mark.skip(reason="CBR-5429")
def test_view_channel_details(caplog, token_user_1, new_channel_with_package):
    channel_details = [
        f"\tname: {new_channel_with_package}",
        "\tdescription:",
        "\tprivacy:",
        "\t# of artifacts:",
        "\t# of downloads:",
        "\t# mirrors:",
        "\t# of subchannels:",
        "\tcreated:",
        "\tupdated:",
    ]

    main.main(["-t", token_user_1, "channel", "--show", new_channel_with_package])
    for detail in channel_details:
        assert detail in caplog.text, caplog.text

    # TODO: Disabled due to CBR-1637
    # package_details = [
    #     f"Total packages matching spec {new_channel_with_package} found:",
    #     "name: numpy",
    #     "# of files:",
    #     "# of downloads: 0",
    #     "license: BSD 3-Clause",
    #     "description",
    # ]
    # main.main(['-t', token_user_1, 'channel', '--list-packages', new_channel_with_package])
    # for detail in package_details:
    #     assert detail in caplog.text, caplog.text

    main.main(["-t", token_user_1, "channel", "--list-files", new_channel_with_package])
    assert "numpy-1.15.4-py37hacdab7b_0.tar.bz2" in caplog.text, caplog.text


@pytest.mark.skip(reason="CBR-5429")
def test_view_subchannel_details(caplog, token_user_1, new_subchannel_with_package):
    subchannel_name = new_subchannel_with_package.split("/")[-1]
    channel_details = [
        f"\tname: {subchannel_name}",
        "\tdescription:",
        "\tprivacy:",
        "\t# of artifacts:",
        "\t# of downloads:",
        "\t# mirrors:",
        "\t# of subchannels:",
        "\tcreated:",
        "\tupdated:",
        "\towners:",
    ]

    main.main(["-t", token_user_1, "channel", "--show", new_subchannel_with_package])

    for detail in channel_details:
        assert detail in caplog.text, caplog.text

    # TODO: Disabled due to CBR-1637
    # package_details = [
    #     f"Total packages matching spec {new_subchannel_with_package} found:",
    #     "name: numpy",
    #     "# of files:",
    #     "# of downloads: 0",
    #     "license: BSD 3-Clause",
    #     "description",
    # ]
    # main.main(['-t', token_user_1, 'channel', '--list-packages', new_subchannel_with_package])
    # for detail in package_details:
    #     assert detail in caplog.text, caplog.text

    main.main(
        ["-t", token_user_1, "channel", "--list-files", new_subchannel_with_package]
    )
    assert "numpy-1.15.4-py37hacdab7b_0.tar.bz2" in caplog.text, caplog.text


@pytest.mark.skip(reason="CBR-5429")
def test_channel_locking(caplog, token_user_1, new_channel):
    main.main(["-t", token_user_1, "channel", "--lock", new_channel])
    assert f"Channel {new_channel} is now locked" in caplog.text, caplog.text

    main.main(["-t", token_user_1, "channel", "--soft-lock", new_channel])
    assert f"Channel {new_channel} is now soft-locked" in caplog.text, caplog.text

    main.main(["-t", token_user_1, "channel", "--unlock", new_channel])
    assert f"Channel {new_channel} is now unlocked" in caplog.text, caplog.text


@pytest.mark.skip(reason="CBR-5429")
def test_subchannel_locking(caplog, token_user_1, new_subchannel):
    main.main(["-t", token_user_1, "channel", "--lock", new_subchannel])
    assert f"Channel {new_subchannel} is now locked" in caplog.text, caplog.text

    main.main(["-t", token_user_1, "channel", "--soft-lock", new_subchannel])
    assert f"Channel {new_subchannel} is now soft-locked" in caplog.text, caplog.text

    main.main(["-t", token_user_1, "channel", "--unlock", new_subchannel])
    assert f"Channel {new_subchannel} is now unlocked" in caplog.text, caplog.text


# TODO: enhance coverage with negative test cases

# Negative test case, finish this
# def test_duplicate_create_channel(token_user_1, monkeypatch):
#     random_number = randint(0, 9999)
#     channel = 'test_channel_{}'.format(random_number)
#
#     # Reset the mock object
#     log_mock = Mock()
#
#     # add a side effect to collect calls to logger.error so it's easier to assert...
#     _errors = []
#     def error_handlers(msg):
#         _errors.append(msg)
#
#     log_mock.error.side_effect = error_handlers
#     monkeypatch.setattr(main.RepoCommand, 'log', log_mock)
#
#     # only check debug if we pass the -v argument
#     # log_mock.debug.assert_any_call(f'Using token {token_user_1} on {test_url}')
#     # log_mock.debug.assert_any_call(f'Creating channel {channel} on {test_url}')
#     #
#     with pytest.raises(errors.RepoCLIError):
#         # check that the same action logs same debug but generates an error instead of info log..
#         main.main(['-t', token_user_1, 'channel', '--create', channel])
#         log_mock.info.assert_any_call(
#             f'Channel {channel} successfully created'
#         )
#
#     error_msg_prefix = f'Error creating {channel} Server responded with status code 409'
#     assert any([error_msg_prefix in err for err in _errors])
#
#     # check that show cmd on that channel show the correct info
#     # Reset the mock object
#     log_mock = Mock()
#     monkeypatch.setattr(main.channel, 'logger', log_mock)
#     main.main(['-t', token_user_1, 'channel', '--show',  channel])
#     msg = '\n'.join(["Channel details:", '', f'name: {channel}', 'description: ', 'privacy: public'])
#     log_mock.info.assert_any_call(msg)

# def test_access_to_locked_channel

# TODO: test duplicate channel creation
