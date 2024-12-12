"""
Manage your Anaconda Server server-side settings

###### Set server-side settings

You can add a site named **site_name** like this:

    conda repo admin --set SETTING_NAME SETTING_VALUE

Currently supported settings are:

    - `user_channel_autocreate` - if True, a user's channel is created automatically on first login

###### Show server-side settings

Yuo can see current settings:

    conda repo admin --show

###### Set anaconda ident settings

You can set anaconda ident settings:

    conda repo admin --ident_enabled
    or
    conda repo admin --ident_disabled

    You can set anaconda ident settings with starting date and tokens:
    conda repo admin --ident-enforce-date 2020-01-01 --ident-tokens client_token,session_token,environment_token,username,hostname,environment,organization

"""
import logging
from argparse import RawDescriptionHelpFormatter

from .. import errors
from ..utils.format import SettingsFormatter
from ..utils.yaml import safe_load
from .base import SubCommandBase

logger = logging.getLogger("repo_cli")


SETTINGS_VALIDATOR = {"user_channel_autocreate": safe_load}


class SubCommand(SubCommandBase):
    name = "admin"

    def main(self):

        if self.args.show:
            self.show_settings()
            return

        if self.args.get:
            self.show_settings(self.args.get)
            return

        if self.args.set:
            self.update_settings(self.args.set)
            return

        if (
            self.args.ident_enabled
            or self.args.ident_disabled
            or self.args.ident_enforce_date
            or self.args.ident_tokens
        ):
            enabled = True if self.args.ident_enabled else None
            if self.args.ident_disabled:
                enabled = False

            self.update_conda_ident_settings(
                enabled,
                self.args.ident_enforce_date if self.args.ident_enforce_date else None,
                self.args.ident_tokens if self.args.ident_tokens else None,
            )
            return

        raise NotImplementedError("Please use command options")

    def show_settings(self, key=None):
        settings = self.api.get_system_settings()
        if key is not None:
            if key not in settings:
                raise errors.RepoCLIError("%s is an unknown admin setting" % key)
            settings = {key: settings[key]}

        self.log.info(SettingsFormatter.format_object_as_list(settings))

    def update_settings(self, args):
        data = {}
        for key, value in args:
            if key not in SETTINGS_VALIDATOR:
                raise errors.RepoCLIError("%s is an unknown admin setting" % key)
            data[key] = SETTINGS_VALIDATOR[key](value)

        settings = self.api.get_system_settings()
        settings.update(data)
        self.api.update_system_settings(settings)

        self.log.info("Anaconda Server settings are updated")

    def update_conda_ident_settings(self, enabled, date=None, tokens_serialize=None):
        settings = self.api.get_system_settings()

        ident_settings = settings.get("anaconda_ident_settings", {})
        if not ident_settings:
            ident_settings = {}

        if enabled is not None:
            ident_settings["enabled"] = enabled
        if date:
            ident_settings["enable_from"] = date
        if tokens_serialize:
            tokens = tokens_serialize.split(",")
            for token in tokens:
                if token not in [
                    "client_token",
                    "session_token",
                    "environment_token",
                    "username",
                    "hostname",
                    "environment",
                    "organization",
                ]:
                    raise errors.RepoCLIError("Invalid token: %s" % token)
            ident_settings["selected_tokens"] = tokens_serialize.split(",")

        settings["anaconda_ident_settings"] = ident_settings

        self.api.update_system_settings(settings)
        self.log.info("Anaconda Ident settings are updated")

    def add_parser(self, subparsers):
        description = "Anaconda Server admin settings"
        parser = subparsers.add_parser(
            "admin",
            help=description,
            description=description,
            epilog=__doc__,
            formatter_class=RawDescriptionHelpFormatter,
        )

        agroup = parser.add_argument_group("actions")

        agroup.add_argument(
            "--set",
            nargs=2,
            action="append",
            default=[],
            help="sets a server setting value: name value",
            metavar=("name", "value"),
        )
        agroup.add_argument("--get", metavar="name", help="get value: name")
        agroup.add_argument(
            "--show", action="store_true", default=False, help="show all variables"
        )

        agroup.add_argument(
            "--ident_enabled",
            action="store_true",
            default=True,
            help="enable anaconda ident",
        )
        agroup.add_argument(
            "--ident_disabled",
            action="store_true",
            default=False,
            help="disable anaconda ident",
        )
        agroup.add_argument(
            "--ident-enforce-date",
            metavar="date",
            help="set anaconda ident date from when conda ident is mandatory",
        )
        agroup.add_argument(
            "--ident_tokens",
            metavar="tokens",
            help="set comma seperated list of tokens for anaconda ident. Must be one of: client_token, session_token, environment_token, username, hostname, environment, organization]",
        )

        parser.set_defaults(main=self.main, sub_parser=parser)
