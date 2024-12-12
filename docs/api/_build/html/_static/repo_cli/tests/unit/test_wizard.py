from unittest.mock import Mock, patch

from repo_cli.commands import wizard
from repo_cli.tests.unit.test_utils.fixtures import fixtures as all_channels

mock_self = Mock()
mock_self.log.warning = Mock()
w = wizard.SubCommand(mock_self)


def test_purge_channels():
    expected_result = ["test"]

    input_value = "test c2 b1"
    channels = w.purge_channels(input_value, all_channels)

    assert channels == expected_result


@patch("builtins.input", side_effect=["c2", "test"])
def test_no_empty_defaults(monkeypatch):
    expected_result = ["test"]

    channels = w.get_default_channels(all_channels)
    # Assert that it failed when called with invalid channel 'c2'
    mock_self.log.warning.assert_called_with(
        "At least one channel should be added to default channels! Try again..."
    )
    assert channels == expected_result
