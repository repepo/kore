# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Anaconda Server login dialogs."""

__all__ = ['TeamEditionSetDomainPage', 'TeamEditionLoginPage', 'TeamEditionAddChannelsPage']

import typing
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget  # pylint: disable=no-name-in-module
from anaconda_navigator.utils import get_domain_from_api_url
from anaconda_navigator.api.anaconda_api import AnacondaAPI
from anaconda_navigator.api.team_edition_api import SSL_ERROR_MESSAGE
from anaconda_navigator.config import AnacondaBrand, CONF
from anaconda_navigator.utils import url_utils
from anaconda_navigator.widgets import ButtonPrimary
from anaconda_navigator.widgets.manager.channels import SelectableChannelsListTable
from anaconda_navigator.static.images import ANACONDA_SERVER_LOGIN_LOGO
from anaconda_navigator.widgets.dialogs.login import styling
from anaconda_navigator.widgets.dialogs.login.base_dialogs import (
    BasePage, BaseSettingPage, TrustServerDialog, BaseLoginPage,
)
from . import utils


TEAM_SET_DOMAIN_TEXT_CONTAINER = utils.TextContainer(
    form_primary_text=(
        'Looks like this is the first time you are logging into Anaconda Server (previously Team Edition). Please '
        'set your Anaconda Server domain.'
    ),
    form_secondary_text='You only need to set this domain once. You can always change this in your Preferences.',
    form_input_label_text='Anaconda Server Domain',
    form_submit_button_text='Set Domain',
    info_frame_logo_path=ANACONDA_SERVER_LOGIN_LOGO,
    info_frame_text=(
        utils.Span(
            'Login to configure Conda and Navigator to install packages from '  # pylint: disable=implicit-str-concat
            'your Anaconda Server (previously Team Edition) instance.'
        ),
    ),
)

TEAM_LOGIN_TEXT_CONTAINER = utils.TextContainer(
    info_frame_logo_path=ANACONDA_SERVER_LOGIN_LOGO,
    info_frame_text=(
        utils.Span(
            'Login to configure Conda and Navigator to install packages from '  # pylint: disable=implicit-str-concat
            'your Anaconda Server (previously Team Edition) instance.'
        ),
    ),
    message_box_error_text='The Anaconda Server API domain is not specified! Please, set in preferences.',
)


class TeamEditionSetDomainPage(BaseSettingPage):  # pylint: disable=missing-class-docstring

    def __init__(self, parent: typing.Optional[QWidget] = None) -> None:
        super().__init__(AnacondaAPI(), TEAM_SET_DOMAIN_TEXT_CONTAINER, parent=parent)
        self.api_url_config_option = 'anaconda_server_api_url'
        self.input_line.setPlaceholderText('http(s)://example.com')
        self.button_apply.clicked.connect(self.set_domain)

    def check_text(self) -> bool:
        text: str = self.input_line.text()
        domain: str = get_domain_from_api_url(text)

        self.input_line.setText(url_utils.join(domain, 'api/system'))
        if super().check_text():
            self.input_line.setText(domain)
            return True

        self.input_line.setText(text)
        return False


class TeamEditionLoginPage(BaseLoginPage):  # pylint: disable=missing-class-docstring

    def __init__(self, parent: typing.Optional[QWidget] = None) -> None:
        super().__init__(AnacondaAPI(), TEAM_LOGIN_TEXT_CONTAINER, parent=parent)
        self.api_url_config_option = 'anaconda_server_api_url'
        self.brand = AnacondaBrand.TEAM_EDITION

    def _finished(self, worker, output, error):
        """
        Callback for the login procedure after worker has finished.

        If success, sets the token, 'username' attribute to parent widget
        and sends the accept signal.

        Otherwise, outputs error messages.
        """
        if output:
            self.accept()
            self.create_login_data()

            # `create_login_data` may overwrite `.condarc` file with backup, so we need to make sure that trusted server
            # is applied in this version of the configuration as well
            if url_utils.netloc(CONF.get('main', 'logged_api_url', '')) in CONF.get('ssl', 'trusted_servers', []):
                self.api.client_set_ssl(False)
            else:
                self.api.client_reset_ssl()

        elif error:
            self.label_message.setText(f'<i>{error}</i>')
            self.label_message.setVisible(True)
            self._track_error()

            if error == SSL_ERROR_MESSAGE:
                if TrustServerDialog(url=CONF.get('main', self.api_url_config_option), parent=self).exec_():
                    self.login()
                    return

            CONF.set_logged_data()
            self.api.client_reload()

        self.button_login.setDisabled(False)
        self.check_text()
        QApplication.restoreOverrideCursor()

    def _track_error(self):
        self.text_password.setFocus()
        self.text_password.selectAll()

    def create_login_data(self):
        self.api.create_login_data()


class TeamEditionAddChannelsPage(BasePage):  # pylint: disable=missing-class-docstring

    def __init__(
            self,
            msg: str = 'Select channels to be used',
            btn_add_msg: str = 'Add Channels',
            btn_cancel_msg: str = 'Skip',
            parent: typing.Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.api = AnacondaAPI()

        QApplication.restoreOverrideCursor()
        pixmap = QPixmap(ANACONDA_SERVER_LOGIN_LOGO)
        pixmap = pixmap.scaledToWidth(200, Qt.SmoothTransformation)
        label_icon = styling.LabelLoginLogo()
        label_icon.setPixmap(pixmap)
        label_icon.setAlignment(Qt.AlignLeft)
        self.label_information = QLabel(msg)

        rc_data = self.api._conda_api.load_rc() or {}
        api_channels_data = self.api.get_channels()
        channels = rc_data.get('channels', [])
        default_channels = rc_data.get('default_channels', [])

        self.channels_table = SelectableChannelsListTable(
            self, table_data=api_channels_data, channels=channels, default_channels=default_channels
        )
        self.channels_table.setMaximumWidth(650)

        self.button_skip = styling.SecondaryButton(btn_cancel_msg)
        self.button_add = ButtonPrimary(btn_add_msg)

        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.button_skip)
        buttons_layout.addWidget(self.button_add)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(label_icon, Qt.AlignLeft)
        self.main_layout.addWidget(self.label_information, Qt.AlignRight)
        self.main_layout.addWidget(self.channels_table, Qt.AlignCenter)
        self.main_layout.addLayout(buttons_layout)

        self.setLayout(self.main_layout)

        self.button_skip.clicked.connect(self.reject)
        self.button_add.clicked.connect(self.add_channels)

    def add_channels(self):  # pylint: disable=missing-function-docstring
        default_channels, channels = self.channels_table.get_selected_channels()

        if not default_channels:
            self.label_information.setText(
                'At least one channel should be added to <b>default_channels</b>! Please add...',
            )
            return

        self.api.update_channels(default_channels, channels)
        self.accept()
