# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Anaconda Professional login dialogs."""

__all__ = ['CommercialEditionLoginPage']

import typing
from qtpy.QtWidgets import QApplication,  QLineEdit, QWidget  # pylint: disable=no-name-in-module
from conda_token.repo_config import CondaTokenError, validate_token, token_set
from anaconda_navigator.api.anaconda_api import AnacondaAPI
from anaconda_navigator.config import CONF, AnacondaBrand
from anaconda_navigator.static.images import ANACONDA_PROFESSIONAL_LOGIN_LOGO
from .base_dialogs import BaseSettingPage
from . import utils


COMMERCIAL_LOGIN_TEXT_CONTAINER = utils.TextContainer(
    form_primary_text=(
        'Looks like this is the first time you are logging into Anaconda Professional (previously Commercial Edition). '
        'Please set your unique access token.'
    ),
    form_secondary_text='You only need to set this token once. You can always change this in your Preferences.',
    form_input_label_text='Enter Anaconda Professional Token',
    form_submit_button_text='Set Token',
    info_frame_logo_path=ANACONDA_PROFESSIONAL_LOGIN_LOGO,
    info_frame_text=(
        utils.Span(
            'Configure Conda and Navigator to install packages from the '  # pylint: disable=implicit-str-concat
            'open-source distribution optimized for commercial use and compliance with our <a href="'
        ),
        utils.UrlSpan(
            'https://www.anaconda.com/terms-of-use/',
            utm_medium='connect-pro',
            utm_content='tos',
        ),
        utils.Span(
            '" style="color:#43B049;text-decoration: none">Terms of Service</a>. Subscription required.'
        ),
    ),
)


class CommercialEditionLoginPage(BaseSettingPage):  # pylint: disable=missing-class-docstring

    def __init__(self, parent: typing.Optional[QWidget] = None) -> None:
        super().__init__(AnacondaAPI(), COMMERCIAL_LOGIN_TEXT_CONTAINER, parent=parent)

        self.button_apply.clicked.connect(self.set_token)
        self.input_line.setEchoMode(QLineEdit.Password)
        self.brand = AnacondaBrand.COMMERCIAL_EDITION

    def set_token(self):  # pylint: disable=missing-function-docstring
        ce_token = self.input_line.text()
        self.label_message.setText('')

        if self.check_text():
            token_set(ce_token)
            commercial_edition_url = CONF.get('main', 'anaconda_professional_url')
            CONF.set_logged_data(commercial_edition_url, self.brand)
            self.api.client_reload()
            self.accept()

        QApplication.restoreOverrideCursor()

    def check_text(self):
        try:
            validate_token(self.input_line.text())
        except CondaTokenError as e:  # pylint: disable=invalid-name
            self.label_message.setText(str(e))
            self.label_message.setVisible(bool(self.input_line.text()))
            return False

        self.button_apply.setEnabled(True)
        self.label_message.setVisible(False)

        return True
