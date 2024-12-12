# -*- coding: utf-8 -*-

# pylint: disable=attribute-defined-outside-init,no-name-in-module

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Base classes of login dialogs."""

__all__ = ['BasePage', 'BaseLoginPage', 'TrustServerDialog', 'BaseSettingPage']

import ast
import contextlib
import typing
from qtpy.QtCore import Qt, QUrl
from qtpy.QtGui import QDesktopServices, QPixmap
from qtpy.QtWidgets import QApplication, QHBoxLayout, QLabel, QLineEdit, QVBoxLayout, QWidget
from anaconda_navigator.api import anaconda_api
from anaconda_navigator.api import download_api
from anaconda_navigator.config import CONF
from anaconda_navigator.utils import attribution
from anaconda_navigator.utils import telemetry
from anaconda_navigator.utils import url_utils
from anaconda_navigator.widgets import ButtonPrimary, ButtonCancel, SpacerVertical
from anaconda_navigator.widgets.dialogs import StaticDialogBase, MessageBoxInformation
from . import styling
from .utils import TextContainer


class BasePage(StaticDialogBase):  # pylint: disable=missing-class-docstring,too-many-instance-attributes

    @property
    def username(self) -> str:
        """Return the logged username."""
        with contextlib.suppress(Exception):
            return self.text_login.text().lower()
        return ''

    def setup(self) -> None:
        """Setup signals and call all common init helpers"""

    def _get_forgot_links_widget(self, text_container: TextContainer) -> typing.Optional[QWidget]:
        if not text_container.form_forgot_links_msg:
            return None

        self.forgot_login_url = None
        self.forgot_password_url = None
        self.forgot_links = QLabel(text_container.form_forgot_links_msg)

        forgot_layout = QHBoxLayout()
        forgot_layout.addWidget(self.forgot_links, 0, Qt.AlignLeft)
        forgot_layout.addStretch(100000000)

        forgot_links_widget = QWidget()
        forgot_links_widget.setLayout(forgot_layout)

        return forgot_links_widget

    def _get_info_frame(self, text_container: TextContainer) -> styling.WidgetLoginInfoFrame:
        info_frame_logo: typing.Optional[styling.LabelLoginLogo] = None
        if text_container.info_frame_logo_path:
            info_frame_logo = self._get_logo_icon(text_container.info_frame_logo_path)

        self.label_information = QLabel(''.join(map(str, text_container.info_frame_text)))
        self.label_information.setWordWrap(True)

        info_layout = QVBoxLayout()

        if info_frame_logo is not None:
            info_layout.addWidget(info_frame_logo)

        info_layout.addWidget(self.label_information)

        info_widget: styling.WidgetLoginInfoFrame = styling.WidgetLoginInfoFrame()
        info_widget.setLayout(info_layout)

        return info_widget

    def _get_form_frame(self, text_container: TextContainer) -> styling.WidgetLoginFormFrame:
        self.label_login = QLabel('Username:')
        self.label_password = QLabel('Password:')

        self.text_login = QLineEdit()

        self.text_password = QLineEdit()
        self.text_password.setEchoMode(QLineEdit.Password)

        self.label_message = styling.LabelMainLoginText('')
        self.label_message.setWordWrap(True)
        self.label_message.setVisible(False)

        self.button_login = ButtonPrimary('Sign In')
        self.button_login.setDefault(True)

        login_form_layout = QVBoxLayout()

        for widget in (self.label_login, self.text_login, self.label_password, self.text_password, self.label_message):
            login_form_layout.addWidget(widget)

        forgot_links_widget = self._get_forgot_links_widget(text_container)
        if forgot_links_widget:
            login_form_layout.addWidget(forgot_links_widget)

        login_form_layout.addWidget(self.button_login, 0, Qt.AlignHCenter)

        login_form_widget = styling.WidgetLoginFormFrame()
        login_form_widget.setLayout(login_form_layout)

        return login_form_widget

    def _get_header_frame(self, text_container: TextContainer) -> typing.Optional[styling.WidgetHeaderFrame]:
        if not text_container.header_frame_logo_path:
            return None

        label_icon: styling.LabelLoginLogo = styling.LabelLoginLogo()
        label_icon.setPixmap(QPixmap(self.text_container.header_frame_logo_path))
        label_icon.setScaledContents(True)  # important on High DPI!
        label_icon.setAlignment(Qt.AlignLeft)

        header_layout = QVBoxLayout()
        header_layout.addWidget(label_icon)

        header_widget = styling.WidgetHeaderFrame()
        header_widget.setLayout(header_layout)

        return header_widget

    def _show_message_box(self, title: str, text: str) -> None:
        msg_box = MessageBoxInformation(title=title, text=text)
        msg_box.exec_()

        self.button_login.setDisabled(False)
        self.check_text()
        QApplication.restoreOverrideCursor()

    @staticmethod
    def _get_logo_icon(logo_path: str) -> styling.LabelLoginLogo:
        label_icon = styling.LabelLoginLogo()
        label_icon.setPixmap(QPixmap(logo_path))
        label_icon.setScaledContents(True)  # important on High DPI!
        label_icon.setAlignment(Qt.AlignLeft)
        return label_icon

    def open_url(self, url: str) -> None:
        """Open given url in the default browser and log the action."""
        QDesktopServices.openUrl(QUrl(url))
        telemetry.ANALYTICS.instance.event('redirect', {'url': str(url)})

    def update_style_sheet(self) -> None:
        """Update custom css style sheet."""


class CommonPage(BasePage):  # pylint: disable=missing-class-docstring

    def __init__(
            self,
            api: anaconda_api._AnacondaAPI,
            text_container: TextContainer,
            parent: typing.Optional[QWidget] = None,
    ) -> None:
        """Initialize new :class:`~CommonPage` instance."""
        super().__init__(parent)

        self.api = api
        self.text_container = text_container

        header_widget = self._get_header_frame(text_container)
        form_widget = self._get_form_frame(text_container)
        info_widget = self._get_info_frame(text_container)

        title = styling.LabelMainLoginTitle(text_container.title or 'Sign in to access your repository')
        title.setWordWrap(True)

        body_layout = QHBoxLayout()
        body_layout.addWidget(info_widget)
        body_layout.addWidget(form_widget)

        body_page_widget = styling.WidgetLoginPageContent()
        body_page_widget.setLayout(body_layout)

        main_layout = QVBoxLayout()
        main_layout.addWidget(title)

        if header_widget:
            main_layout.addWidget(header_widget)
        main_layout.addWidget(body_page_widget)

        self.setLayout(main_layout)
        self.setup()


class BaseLoginPage(CommonPage):  # pylint: disable=missing-class-docstring

    def __init__(
            self,
            api: anaconda_api._AnacondaAPI,
            text_container: TextContainer,
            parent: typing.Optional[QWidget] = None,
    ) -> None:
        """Initialize new :class:`~BaseLoginPage` instance."""
        super().__init__(api, text_container, parent=parent)

    def setup(self):
        attribution.UPDATER.instance.sig_updated.connect(self.update_links)
        self.update_links()

        self.text_login.setFocus()
        self.text_login.textEdited.connect(self.check_text)
        self.text_password.textEdited.connect(self.check_text)
        self.button_login.clicked.connect(self.login)

        self.check_text()
        self.update_style_sheet()

    def check_text(self):
        """Check that `login` and `password` are not empty strings.

        If not empty and disable/enable buttons accordingly.
        """
        login = self.text_login.text()
        password = self.text_password.text()

        self.button_login.setDisabled(not (login and password))

    def login(self):  # pylint: disable=missing-function-docstring
        api_url = CONF.get('main', self.api_url_config_option)
        if not api_url:
            self._show_message_box(
                title='Login Error',
                text=self.text_container.domain_not_found_msg,
            )
            return

        username_text = self.text_login.text().lower()
        self.button_login.setEnabled(False)
        self.label_message.setText('')
        self.text_login.setText(username_text)

        QApplication.setOverrideCursor(Qt.WaitCursor)

        # Reload the client to the other one, if needed.
        CONF.set_logged_data(api_url, self.brand)

        # Disable SSL verification in clients for trusted server
        verify_ssl: typing.Optional[bool] = None
        if url_utils.netloc(api_url) in CONF.get('ssl', 'trusted_servers', []):
            self.api.client_set_ssl(False)
            verify_ssl = False
        else:
            self.api.client_reset_ssl()

        self.api.client_reload()

        worker = self.api.login(username_text, self.text_password.text(), verify_ssl=verify_ssl)
        worker.sig_finished.connect(self._finished)

    def _finished(self, worker, output, error):  # pylint: disable=unused-argument
        """
        Callback for the login procedure after worker has finished.

        If success, sets the token, 'username' attribute to parent widget
        and sends the accept signal.

        Otherwise, outputs error messages.
        """
        token = output
        username = self.text_login.text().lower()

        if token:
            self.accept()
            self.create_login_data()

            # `create_login_data` may overwrite `.condarc` file with backup, so we need to make sure that trusted server
            # is applied in this version of the configuration as well
            if url_utils.netloc(CONF.get('main', 'logged_api_url', '')) in CONF.get('ssl', 'trusted_servers', []):
                self.api.client_set_ssl(False)
            else:
                self.api.client_reset_ssl()

        elif error:
            CONF.set_logged_data()

            # after failed login attempt - restore previous settings
            self.api.client_reset_ssl()

            bold_username = f'<b>{username}</b>'

            # The error might come in (error_message, http_error) format
            try:
                error_message = ast.literal_eval(str(error))[0]
            except Exception:  # cov-skip ; pylint: disable=broad-except
                error_message = str(error)

            error_message = error_message.lower().capitalize()
            error_message = error_message.split(', ')[0]
            error_text = f'<i>{error_message}</i>'
            error_text = error_text.replace(username, bold_username)
            self.label_message.setText(error_text)
            self.label_message.setVisible(True)

            if error_message:
                self.text_password.setFocus()
                self.text_password.selectAll()

        self.button_login.setDisabled(False)
        self.check_text()
        QApplication.restoreOverrideCursor()

    def update_links(self):
        """Fill with urls placeholders within text on page."""

    def create_login_data(self):
        """Post login configurations"""


class TrustServerDialog(StaticDialogBase):  # pylint: disable=too-many-instance-attributes
    """
    Dialog to verify adding `url` to trusted servers.

    :param url: URL being verified.
    :param title: Initial value for dialog title.
    :param message: Initial value for dialog message.
    :param accept_text: Initial value for confirmation button text.
    :param cancel_text: Initial value for cancel button text.
    :param parent: Parent window for this dialog.
    """

    def __init__(  # pylint: disable=too-many-arguments
            self,
            url: str,
            title: str = 'SSL Verification Failed!',
            message: str = (
                'It looks like this server is using an SSL certificate which we cannot verify.\n\n'
                'Do you want to trust this server anyway?'
            ),
            accept_text: str = 'Trust Server and Continue',
            cancel_text: str = 'Cancel',
            parent: typing.Optional[QWidget] = None,
    ) -> None:
        """Initialize new :class:`~SSLCheckDialog` instance."""
        super().__init__(parent)

        self.__url: typing.Final[str] = url

        self._title_label = styling.LabelMainLoginTitle(title)
        self._title_label.setWordWrap(True)

        self._message_label = QLabel(message)
        self._message_label.setWordWrap(True)

        self._accept_button = ButtonPrimary(accept_text)
        self._accept_button.setDefault(True)
        self._accept_button.clicked.connect(self.__accept)

        self._cancel_button = ButtonCancel(cancel_text)
        self._cancel_button.clicked.connect(self.reject)

        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self._cancel_button)
        self.button_layout.addWidget(SpacerVertical())
        self.button_layout.addWidget(self._accept_button)

        self.button_line = QWidget()
        self.button_line.setLayout(self.button_layout)

        self._content_layout = QVBoxLayout()
        self._content_layout.addWidget(self._message_label)
        self._content_layout.addWidget(self.button_line, 0, Qt.AlignLeft)

        self._content_frame = styling.WidgetNoticeFrame()
        self._content_frame.setLayout(self._content_layout)

        self._main_layout: QVBoxLayout = QVBoxLayout()
        self._main_layout.addWidget(self._title_label)
        self._main_layout.addWidget(self._content_frame)

        self.setLayout(self._main_layout)
        self.update_style_sheet()

    @property
    def url(self) -> str:  # noqa: D401
        """URL being validated."""
        return self.__url

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
    def accept_text(self) -> str:  # noqa: D401
        """Text of the confirmation button."""
        return self._accept_button.text()

    @accept_text.setter
    def accept_text(self, value: str) -> None:
        """Update `accept_text` value."""
        self._accept_button.setText(value)

    @property
    def cancel_text(self) -> str:  # noqa: D401
        """Text of the cancel button."""
        return self._cancel_button.text()

    @cancel_text.setter
    def cancel_text(self, value: str) -> None:
        """Update `cancel_text` value."""
        self._cancel_button.setText(value)

    def __accept(self) -> None:
        """Accept trusting the URL."""
        domain: str = url_utils.netloc(self.__url)
        trusted_servers: typing.List[str] = CONF.get('ssl', 'trusted_servers', [])
        if domain not in trusted_servers:
            trusted_servers.append(domain)
            CONF.set('ssl', 'trusted_servers', trusted_servers)

        self.accept()


class BaseSettingPage(CommonPage):  # pylint: disable=missing-class-docstring

    def __init__(
            self,
            api: anaconda_api._AnacondaAPI,
            text_container: TextContainer,
            parent: typing.Optional[QWidget] = None,
    ) -> None:
        """Initialize new :class:`~BaseSettingPage` instance."""
        super().__init__(api, text_container, parent=parent)

    def setup(self):
        self.label_information.linkActivated.connect(self.open_url)
        self.update_style_sheet()
        self.input_line.setFocus()

    def _get_form_frame(self, text_container):
        self.label_text = styling.LabelMainLoginText(text_container.form_primary_text)
        self.label_note = styling.LabelMainLoginSubTitle(text_container.form_secondary_text)
        self.input_label = QLabel(text_container.form_input_label_text)
        self.label_text.setWordWrap(True)
        self.label_note.setWordWrap(True)

        self.input_line = QLineEdit()

        self.label_message = styling.LabelMainLoginText('')
        self.label_message.setWordWrap(True)
        self.label_message.setVisible(False)

        self.button_apply = ButtonPrimary(text_container.form_submit_button_text)
        self.button_apply.setEnabled(True)
        self.button_apply.setDefault(True)

        login_form_widget = styling.WidgetLoginFormFrame()
        login_form_layout = QVBoxLayout()
        for widget in (self.label_text, self.label_note, self.input_label, self.input_line, self.label_message):
            login_form_layout.addWidget(widget)

        login_form_layout.addWidget(self.button_apply, 0, Qt.AlignHCenter)
        login_form_widget.setLayout(login_form_layout)

        return login_form_widget

    def set_domain(self):  # pylint: disable=missing-function-docstring
        self.input_line.setText(self.input_line.text().lower())
        self.label_message.setText('')

        if self.check_text():
            CONF.set('main', self.api_url_config_option, self.input_line.text().strip('/'))
            self.api.client_reload()
            self.accept()

        QApplication.restoreOverrideCursor()

    def check_text(self) -> bool:  # pylint: disable=missing-function-docstring
        error: typing.Optional[str] = self.is_valid_api(self.input_line.text().lower())
        if error:
            self.label_message.setText(error)
            self.label_message.setVisible(bool(self.input_line.text()))
            return False

        self.button_apply.setEnabled(True)
        self.label_message.setVisible(False)
        return True

    def is_valid_api(
            self,
            url: str,
            verify: typing.Optional[bool] = None,
            allow_blank: bool = False,
    ) -> typing.Optional[str]:
        """Check if a given URL is a valid anaconda api endpoint."""
        if (verify is None) and (url_utils.netloc(url) in CONF.get('ssl', 'trusted_servers', [])):
            verify = False

        check = self.api.download_is_valid_api_url(url, non_blocking=False, verify=verify, allow_blank=allow_blank)
        if check:
            return None

        if (check is download_api.ErrorDetail.ssl_error) and TrustServerDialog(url=url, parent=self).exec_():
            check = self.api.download_is_valid_api_url(url, non_blocking=False, verify=False, allow_blank=allow_blank)
            if check:
                return None

        return 'Invalid API url. Check the url is valid and corresponds to the api endpoint.'
