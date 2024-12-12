# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Anaconda.org login dialogs."""

__all__ = ['AnacondaLoginPage']

import typing
from qtpy import QtWidgets
from anaconda_navigator.api.anaconda_api import AnacondaAPI
from anaconda_navigator.config import CONF, AnacondaBrand
from anaconda_navigator.static.images import ANACONDA_ORG_EDITION_LOGIN_LOGO
from anaconda_navigator.utils import attribution
from anaconda_navigator.utils import url_utils
from .base_dialogs import BaseLoginPage
from . import utils


ANACONDA_LOGIN_TEXT_CONTAINER = utils.TextContainer(
    info_frame_text=(
        utils.Span(
            'Log into Anaconda.org to access private channels and packages. '  # pylint: disable=implicit-str-concat
            'If you don’t have an account, click <a href="{}" style="color:#43B049;text-decoration: none">here</a>.'
        ),
    ),
    form_forgot_links_msg=(
        'Forget your '  # pylint: disable=implicit-str-concat
        '<a href="{username_url}" style="color:#43B049; text-decoration:none">username</a> or '
        '<a href="{password_url}" style="color:#43B049; text-decoration:none">password</a>?'
    ),
    message_box_error_text='The Anaconda.Org API domain is not specified! Please, set in preferences.',
    info_frame_logo_path=ANACONDA_ORG_EDITION_LOGIN_LOGO,
)


class AnacondaLoginPage(BaseLoginPage):  # pylint: disable=missing-class-docstring

    def __init__(self, parent: typing.Optional[QtWidgets.QWidget] = None) -> None:
        """Login dialog."""
        super().__init__(AnacondaAPI(), ANACONDA_LOGIN_TEXT_CONTAINER, parent=parent)
        self.api_url_config_option = 'anaconda_api_url'
        self.brand = AnacondaBrand.ANACONDA_ORG
        self.text_login.setValidator(utils.USER_RE_VALIDATOR)

    def update_links(self):
        """Update links."""
        anaconda_api_url = CONF.get('main', 'anaconda_api_url', self.api.client_get_api_url())

        if not anaconda_api_url:
            return

        base_url = anaconda_api_url.lower().replace('//api.', '//')

        parts = base_url.lower().split('/')
        base_url = '/'.join(parts[:-1]) if parts[-1] == 'api' else base_url

        info_updated_text = self.label_information.text().format(
            attribution.POOL.settings.inject_url_parameters(
                base_url,
                utm_medium='connect-org',
                utm_content='signup',
            )
        )
        forgot_links_updated_text = self.forgot_links.text().format(
            username_url=attribution.POOL.settings.inject_url_parameters(
                url_utils.join(base_url, utils.FORGOT_LOGIN_URL),
                utm_medium='connect-org',
                utm_content='username-restore',
            ),
            password_url=attribution.POOL.settings.inject_url_parameters(
                url_utils.join(base_url, utils.FORGOT_PASSWORD_URL),
                utm_medium='connect-org',
                utm_content='password-reset',
            )
        )

        self.label_information.setText(info_updated_text)
        self.forgot_links.setText(forgot_links_updated_text)

        self.label_information.linkActivated.connect(self.open_url)
        self.forgot_links.linkActivated.connect(self.open_url)
