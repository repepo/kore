"""
Authenticate a user
"""
from __future__ import unicode_literals

import getpass
import logging
import platform
import re
import sys
import webbrowser

from six.moves import input

from .. import errors
from ..utils.config import OIDC_CLIENT_ID, get_config, store_token
from ..utils.local_auth_server import WebServer
from .base import SubCommandBase

logger = logging.getLogger("repo_cli")


class SubCommand(SubCommandBase):
    name = "login"
    manages_auth = True

    def main(self):
        self.login()

    def login(self):
        token = self.interactive_get_token()
        store_token(token, self.args)
        is_admin = self.api.is_admin_jwt
        msg = "login as ADMIN successful" if is_admin else "login successful"
        self.log.info(msg)
        return token  # ['id']

    def get_login_and_password(self):
        if getattr(self.args, "login_username", None):
            username = self.args.login_username
        else:
            username = input("Username: ")
        self.username = username
        password = getattr(self.args, "login_password", None)
        return username, password

    def interactive_get_token(self):
        config = get_config()
        if config.get("oauth2"):
            token = self.oauth2_get_token()
        else:
            token = self.direct_get_token()
        return token

    def get_and_validate_user_token(self, force_scopes=None):
        """
        Returns user opaque token

        Args:
            force_scopes (bool): True if old token should be recreated with new scopes
        """
        user_token = self.api.get_user_token(force_scopes=force_scopes)

        if not user_token:
            msg = "Unable to request the user token. Server was unable to return any valid user token!"
            logger.error(msg)
            raise errors.RepoCLIError(msg)

        return user_token

    def direct_get_token(self):
        username, password = self.get_login_and_password()
        for _ in range(3):
            try:
                if password is None:
                    password = getpass.getpass(stream=sys.stderr)

                self.api.login(username, password)
                user_token = self.get_and_validate_user_token(
                    force_scopes=self.args.force_scopes
                    if "force_scopes" in self.args
                    else None
                )
                return user_token

            except errors.Unauthorized:
                logger.error("Invalid Username password combination, please try again")
                password = None
                continue

        raise errors.RepoCLIError("You've reached maximum login attempts")

    def get_openid_configuration_url(self):
        url = self.api.get_authorize_url()
        matches = re.match("(/auth/realms/([a-z][a-z-]+)/).*", url.path)
        if not matches:
            logger.info(f"Auth path is not supported: {url.path}")
            raise errors.WrongRepoAuthSetup()

        url = url._replace(path=matches[1] + ".well-known/openid-configuration")
        return url.geturl()

    def oauth2_get_token(self):
        openid_configuration_url = self.get_openid_configuration_url()
        self.log.debug(f"OpenID configuration: {openid_configuration_url}")
        server = WebServer(
            client_id=OIDC_CLIENT_ID,
            openid_configuration_url=openid_configuration_url,
            verify_ssl=self.verify_ssl,
        )
        thread = server.start()
        thread.start()
        self.log.info("Opening browser to get a token...")
        webbrowser.open(server.localhost_url())
        thread.join(120)
        if not server.access_token:
            msg = "Unable to request the user token. Server was unable to return any valid token!"
            logger.error(msg)
            raise errors.RepoCLIError(msg)
        self.api._jwt = server.access_token
        user_token = self.get_and_validate_user_token(
            force_scopes=self.args.force_scopes
        )
        return user_token

    def add_parser(self, subparsers):
        self.subparser = subparser = subparsers.add_parser(
            "login", help="Authenticate a user", description=__doc__
        )
        subparser.add_argument(
            "--hostname",
            default=platform.node(),
            help="Specify the host name of this login, this should be unique (default: %(default)s)",
        )
        subparser.add_argument(
            "--username",
            dest="login_username",
            help="Specify your username. If this is not given, you will be prompted",
        )
        subparser.add_argument(
            "--password",
            dest="login_password",
            help="Specify your password. If this is not given, you will be prompted",
        )
        subparser.add_argument(
            "--force-scopes",
            dest="force_scopes",
            action="store_true",
            help="Previous token will be recreated with actual user scopes",
        )
        subparser.set_defaults(main=self.main)
