# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Enterprise Edition login dialogs."""

__all__ = ['NoticePage', 'EnterpriseRepoSetDomainPage', 'EnterpriseRepoLoginPage']

import typing
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QVBoxLayout, QWidget  # pylint: disable=no-name-in-module
from anaconda_navigator.api.anaconda_api import AnacondaAPI
from anaconda_navigator.config import AnacondaBrand
from anaconda_navigator.widgets import ButtonPrimary
from anaconda_navigator.widgets.dialogs import StaticDialogBase
from anaconda_navigator.static.images import ANACONDA_ENTERPRISE_LOGIN_LOGO
from .base_dialogs import BaseLoginPage, BaseSettingPage
from . import styling
from . import utils


ENTERPRISE_LOGIN_TEXT_CONTAINER = utils.TextContainer(
    info_frame_text=(
        utils.Span(
            'Login to configure Conda and Navigator to install packages from your on-premise package repository.'
        ),
    ),
    message_box_error_text='The Enterprise 4 Repository API domain is not specified! Please, set in preferences.',
    info_frame_logo_path=ANACONDA_ENTERPRISE_LOGIN_LOGO,
)

ENTERPRISE_SET_DOMAIN_TEXT_CONTAINER = utils.TextContainer(
    form_primary_text=(
        'Looks like this is the first time you are logging into Enterprise 4 Repository. Please set your Enterprise '
        '4 Repository API domain.'
    ),
    form_secondary_text='You only need to set this domain once. You can always change this in your Preferences.',
    form_input_label_text='Enter Enterprise 4 Repository API Domain',
    form_submit_button_text='Set Domain',
    info_frame_logo_path=ANACONDA_ENTERPRISE_LOGIN_LOGO,
    info_frame_text=(
        utils.Span(
            'Login to configure Conda and Navigator to install packages from your on-premise package repository.'
        ),
    )
)


class NoticePage(StaticDialogBase):
    """
    Simple notification popup, that might be integrated into login process.

    :param title: Initial value for notification title.
    :param message: Initial value for notification message.
    :param button_text: Initial value for confirmation button text.

                        If it is not provided - login dialog will be closed ('accepted').
    """

    def __init__(
            self,
            title: str = 'You\'re almost ready!',
            message: str = 'Please verify you have the correct channels configured before proceeding',
            button_text: str = 'OK',
            parent: typing.Optional[QWidget] = None,
    ) -> None:
        """Initialize new :class:`~BaseNoticePage` instance."""
        super().__init__(parent)

        self._title_label = styling.LabelMainLoginTitle(title)
        self._title_label.setWordWrap(True)

        self._message_label = QLabel(message)
        self._message_label.setWordWrap(True)

        self._accept_button = ButtonPrimary(button_text)
        self._accept_button.setDefault(True)
        self._accept_button.clicked.connect(self.accept)

        self._content_layout = QVBoxLayout()
        self._content_layout.addWidget(self._message_label)
        self._content_layout.addWidget(self._accept_button, 0, Qt.AlignHCenter)

        self._content_frame = styling.WidgetNoticeFrame()
        self._content_frame.setLayout(self._content_layout)

        self._main_layout: QVBoxLayout = QVBoxLayout()
        self._main_layout.addWidget(self._title_label)
        self._main_layout.addWidget(self._content_frame)

        self.setLayout(self._main_layout)
        self.update_style_sheet()

    @property
    def title(self) -> str:  # noqa: D401
        """Title of the notification."""
        return self._title_label.text()

    @title.setter
    def title(self, value: str) -> None:
        """Update `title` value."""
        self._title_label.setText(value)

    @property
    def message(self) -> str:  # noqa: D401
        """Main message of the notification."""
        return self._title_label.text()

    @message.setter
    def message(self, value: str) -> None:
        """Update `message` value."""
        self._title_label.setText(value)

    @property
    def button_text(self) -> str:  # noqa: D401
        """Text of the confirmation button."""
        return self._accept_button.text()

    @button_text.setter
    def button_text(self, value: str) -> None:
        """Update `button_text` value."""
        self._accept_button.setText(value)


class EnterpriseRepoSetDomainPage(BaseSettingPage):  # pylint: disable=missing-class-docstring

    def __init__(self, parent: typing.Optional[QWidget] = None) -> None:
        """Initialize new :class:`~EnterpriseRepoSetDomainPage` instance."""
        super().__init__(AnacondaAPI(), ENTERPRISE_SET_DOMAIN_TEXT_CONTAINER, parent=parent)
        self.api_url_config_option = 'enterprise_4_repo_api_url'
        self.input_line.setPlaceholderText('http(s)://example.com')
        self.button_apply.clicked.connect(self.set_domain)


class EnterpriseRepoLoginPage(BaseLoginPage):  # pylint: disable=missing-class-docstring

    def __init__(self, parent: typing.Optional[QWidget] = None) -> None:
        """Initialize new :class:`~EnterpriseRepoLoginPage` instance."""
        super().__init__(AnacondaAPI(), ENTERPRISE_LOGIN_TEXT_CONTAINER, parent=parent)
        self.api_url_config_option = 'enterprise_4_repo_api_url'
        self.brand = AnacondaBrand.ENTERPRISE_EDITION
        self.text_login.setValidator(utils.USER_RE_VALIDATOR)

    def create_login_data(self):
        """
        Backup actual .condarc and replace it with .condarc duplicate with empty channels list.
        """
        self.api.create_login_data()
