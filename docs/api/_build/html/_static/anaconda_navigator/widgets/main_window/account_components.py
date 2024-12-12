# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Components for user accounts management."""

from __future__ import annotations

__all__ = ['AccountsComponent']

import time
import typing
from urllib import parse

from conda_token import repo_config
from conda_token.repo_config import configure_default_channels, token_list
from qtpy import QtCore
from qtpy import QtWidgets, QtGui
from repo_cli.utils.config import get_config, load_token

from anaconda_navigator.api import cloud
from anaconda_navigator import config
from anaconda_navigator.config import preferences
from anaconda_navigator.static.images import EXCLAMATION_CIRCLE_PATH
from anaconda_navigator.utils import telemetry
from anaconda_navigator import widgets
from anaconda_navigator.widgets.dialogs import login as login_dialogs
from anaconda_navigator.widgets.dialogs.login import TeamEditionAddChannelsPage
from . import common

if typing.TYPE_CHECKING:
    from anaconda_navigator.utils import workers
    from anaconda_navigator.widgets import main_window
    from anaconda_navigator.widgets.dialogs.login import account_dialogs


class ButtonLabelLogin(QtWidgets.QLabel):  # pylint: disable=too-few-public-methods
    """Button used in CSS styling."""


class ButtonLogin(widgets.ButtonPrimary):
    """Button used in CSS styling."""


class AccountsComponent(common.Component):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Component for account management."""

    __alias__ = 'accounts'

    def __init__(self, parent: 'main_window.MainWindow') -> None:
        """Initialize new :class:`~AccountsComponent` instance."""
        super().__init__(parent=parent)

        self.__authenticated: bool = False
        self.__brand: typing.Optional[str] = config.AnacondaBrand.DEFAULT
        self.__token: typing.Optional[str] = self.main_window.api._client_api.load_token()
        self.__username: str = ''

        self.__timer: typing.Final[QtCore.QTimer] = QtCore.QTimer()
        self.__timer.setInterval(5000)
        self.__timer.timeout.connect(self.__check_for_new_login)

        self.__account_label: typing.Final[QtWidgets.QLabel] = ButtonLabelLogin()
        self.__account_label.setFocusPolicy(QtCore.Qt.NoFocus)
        self.__account_label.setText('')
        self.__account_label.linkActivated.connect(self.main_window.open_url)
        self.__account_label_icon = QtWidgets.QLabel()
        self.__account_label_icon.setVisible(False)

        self.__account_label_layout = QtWidgets.QHBoxLayout()
        self.__account_label_layout.addWidget(self.__account_label_icon, alignment=QtCore.Qt.AlignCenter)
        self.__account_label_layout.addSpacing(4)
        self.__account_label_layout.addWidget(self.__account_label, alignment=QtCore.Qt.AlignLeft)
        self.__account_label_widget = QtWidgets.QWidget()
        self.__account_label_widget.setLayout(self.__account_label_layout)

        self.__login_button: typing.Final[QtWidgets.QPushButton] = ButtonLogin()
        self.__login_button.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.__login_button.setDefault(True)
        self.__login_button.setText('Connect')
        self.__login_button.clicked.connect(self.__show_accounts)

    @property
    def username(self) -> str:
        """Login of logged in user."""
        return self.__username

    @property
    def account_label(self) -> QtWidgets.QLabel:
        """Label with details of account login."""
        return self.__account_label

    @property
    def account_label_widget(self) -> QtWidgets.QWidget:
        """Widget with icon and title about details of account login."""
        return self.__account_label_widget

    @property
    def login_button(self) -> QtWidgets.QPushButton:
        """Button to trigger login action."""
        return self.__login_button

    def setup(self, worker: typing.Any, output: typing.Any, error: str, initial: bool) -> None:
        """Perform component configuration from `conda_data`."""
        if self.__brand == config.AnacondaBrand.TEAM_EDITION and initial:
            TeamEditionAddChannelsPage().exec()

    def update_login_status(self, user_data=None):
        """Update login button and information."""
        if self.main_window.config.get('main', 'logged_api_url') or user_data:
            self.__username = user_data.get('login', '') if user_data else self.__username
            self.__authenticated = True

        self.__update_account_label_text()

        # See: https://github.com/ContinuumIO/navigator/issues/1325
        self.main_window.api.client_reload()

        def apply_api_urls(worker, output, error):  # pylint: disable=unused-argument
            if output:
                self.__brand = output.get('brand', config.AnacondaBrand.DEFAULT)
            else:
                self.__brand = config.AnacondaBrand.DEFAULT

            try:
                self.login_button.setEnabled(True)
            except RuntimeError:
                pass  # On CI: wrapped C/C++ object of type ButtonLinkLogin has been deleted

        self.login_button.setEnabled(False)
        worker = self.main_window.api.api_urls()
        worker.username = self.__username
        worker.sig_chain_finished.connect(apply_api_urls)
        QtWidgets.QApplication.restoreOverrideCursor()

    def show_error_icon(self, tooltip: typing.Optional[str] = None) -> None:
        """Show error icon near account_label text"""
        self.__account_label_icon.setPixmap(QtGui.QPixmap(EXCLAMATION_CIRCLE_PATH))
        self.__account_label_icon.setToolTip(tooltip or '')
        self.__account_label_icon.setVisible(True)
        self.__update_account_label_text()

    def hide_error_icon(self) -> None:
        """Show error icon near account_label text"""
        self.__account_label_icon.setToolTip('')
        self.__account_label_icon.setVisible(False)
        self.__update_account_label_text()

    def __update_account_label_text(self) -> None:
        """Update login button and information."""
        result: typing.List[str] = []

        cloud_account: typing.Optional[str] = cloud.CloudAPI().username
        if cloud_account:
            result.append('Cloud')

        if self.__authenticated:
            brand: typing.Optional[str]
            brand, _ = self.main_window.config.get_logged_data()
            result.append(brand or '')

        content: str = ''
        if result:
            if self.__account_label_icon.isVisible():
                content += 'Partially connected to '
            else:
                content += 'Connected to '
            content += ', '.join(
                f'<a href="" style="color:#43B049;text-decoration:none">{item}</a>' for item in result
            )

        self.account_label.setText(content)
        self.account_label.setVisible(bool(content))

    def __show_accounts(self):
        """Open up login dialog or log out depending on logged status."""
        states: 'account_dialogs.AccountStateMapping' = {}

        cloud_username: typing.Optional[str] = cloud.CloudAPI().username
        if cloud_username:
            states['cloud'] = login_dialogs.AccountState(
                status=login_dialogs.AccountStatus.ACTIVE,
                username=cloud_username,
            )
        else:
            states['cloud'] = login_dialogs.AccountState(status=login_dialogs.AccountStatus.AVAILABLE)

        if self.__authenticated:
            brand: typing.Optional[str]
            brand, _ = self.main_window.config.get_logged_data()
            if brand == config.AnacondaBrand.ANACONDA_ORG:
                states['individual'] = login_dialogs.AccountState(
                    status=login_dialogs.AccountStatus.ACTIVE,
                    username=self.__username,
                )
            elif brand == config.AnacondaBrand.COMMERCIAL_EDITION:
                states['commercial'] = login_dialogs.AccountState(
                    status=login_dialogs.AccountStatus.ACTIVE,
                    username=self.__username,
                )
            elif brand == config.AnacondaBrand.TEAM_EDITION:
                states['team'] = login_dialogs.AccountState(
                    status=login_dialogs.AccountStatus.ACTIVE,
                    username=self.__username,
                )
            elif brand == config.AnacondaBrand.ENTERPRISE_EDITION:
                states['enterprise'] = login_dialogs.AccountState(
                    status=login_dialogs.AccountStatus.ACTIVE,
                    username=self.__username,
                )
        else:
            states['individual'] = login_dialogs.AccountState(status=login_dialogs.AccountStatus.AVAILABLE)
            states['commercial'] = login_dialogs.AccountState(status=login_dialogs.AccountStatus.AVAILABLE)
            states['team'] = login_dialogs.AccountState(status=login_dialogs.AccountStatus.AVAILABLE)
            states['enterprise'] = login_dialogs.AccountState(status=login_dialogs.AccountStatus.AVAILABLE)

        selector = login_dialogs.AccountsDialog(
            parent=self.main_window,
            anchor=self.login_button,
            states=states,
        )
        selector.sig_accepted.connect(self.__process_accounts)
        selector.show()

    def __process_accounts(self, outcome: login_dialogs.AccountOutcome, value: login_dialogs.AccountValue) -> None:
        """"""
        if outcome == login_dialogs.AccountOutcome.REJECT:
            return

        if outcome == login_dialogs.AccountOutcome.LOGIN_REQUEST:
            login_functions: typing.Mapping[login_dialogs.AccountValue, typing.Callable[[], typing.Any]] = {
                login_dialogs.AccountValue.CLOUD: self.log_into_cloud,
                login_dialogs.AccountValue.INDIVIDUAL_EDITION: self.log_into_individual_edition,
                login_dialogs.AccountValue.COMMERCIAL_EDITION: self.log_into_commercial_edition,
                login_dialogs.AccountValue.TEAM_EDITION: self.log_into_team_edition,
                login_dialogs.AccountValue.ENTERPRISE_EDITION: self.log_into_enterprise_edition,
            }
            login_functions[value]()
            return

        if outcome == login_dialogs.AccountOutcome.LOGOUT_REQUEST:
            logout_functions: typing.Mapping[login_dialogs.AccountValue, typing.Callable[[], typing.Any]] = {
                login_dialogs.AccountValue.CLOUD: self.log_out_from_cloud,
                login_dialogs.AccountValue.INDIVIDUAL_EDITION: self.log_out_from_repository,
                login_dialogs.AccountValue.COMMERCIAL_EDITION: self.log_out_from_repository,
                login_dialogs.AccountValue.TEAM_EDITION: self.log_out_from_repository,
                login_dialogs.AccountValue.ENTERPRISE_EDITION: self.log_out_from_repository,
            }
            logout_functions[value]()
            return

        raise ValueError('Unexpected login outcome')

    def log_into_cloud(self) -> None:
        """Open dialogs to log into Anaconda Cloud."""
        if login_dialogs.CloudLoginPage(parent=self.main_window).exec_():
            self.__update_account_label_text()

    def log_into_individual_edition(self) -> None:
        """Open dialogs to log into Anaconda Individual Edition."""
        credentials_dialog: typing.Final[QtWidgets.QDialog] = login_dialogs.AnacondaLoginPage(
            parent=self.main_window,
        )
        credentials_dialog.exec_()
        self.__postprocess_dialog(dialog=credentials_dialog, edition='individual')

    def log_into_commercial_edition(self) -> None:
        """Open dialogs to log into Anaconda Professional."""
        credentials_dialog: typing.Final[QtWidgets.QDialog] = login_dialogs.CommercialEditionLoginPage(
            parent=self.main_window,
        )
        credentials_dialog.exec_()
        self.__postprocess_dialog(dialog=credentials_dialog, edition='professional')

    def log_into_team_edition(self) -> None:
        """Open dialogs to log into Anaconda Server."""
        if not self.main_window.config.get('main', 'anaconda_server_api_url'):
            domain_dialog: typing.Final[QtWidgets.QDialog] = login_dialogs.TeamEditionSetDomainPage(
                parent=self.main_window,
            )
            if not domain_dialog.exec_():
                return

        credentials_dialog: typing.Final[QtWidgets.QDialog] = login_dialogs.TeamEditionLoginPage(
            parent=self.main_window,
        )
        if not credentials_dialog.exec_():
            return

        login_dialogs.TeamEditionAddChannelsPage(parent=self.main_window).exec_()
        self.__postprocess_dialog(credentials_dialog, edition='server')

    def log_into_enterprise_edition(self) -> None:
        """Open dialogs to log into Anaconda Enterprise Edition."""
        if not self.main_window.config.get('main', 'enterprise_4_repo_api_url'):
            domain_dialog: typing.Final[QtWidgets.QDialog] = login_dialogs.EnterpriseRepoSetDomainPage(
                parent=self.main_window,
            )
            if not domain_dialog.exec_():
                return

        credentials_dialog: typing.Final[QtWidgets.QDialog] = login_dialogs.EnterpriseRepoLoginPage(
            parent=self.main_window,
        )
        if not credentials_dialog.exec_():
            return

        login_dialogs.NoticePage(parent=self.main_window).exec_()
        self.__postprocess_dialog(credentials_dialog, edition='enterprise')

    def log_out_from_cloud(self) -> None:
        """Log out from Anaconda Cloud."""
        cloud.CloudAPI().logout()
        self.__update_account_label_text()

        welcome_state: int = config.CONF.get('internal', 'welcome_state', preferences.WELCOME_DELAYS.first)
        if welcome_state >= preferences.WELCOME_DELAYS.first:
            welcome_state = preferences.WELCOME_DELAYS.last
            config.CONF.set('internal', 'welcome_state', welcome_state)
            config.CONF.set('internal', 'welcome_ts', int(time.time()) + preferences.WELCOME_DELAYS.get(welcome_state))

    def log_out_from_repository(self) -> None:
        """Log out from all repositories (Individual, Professional, Server and Enterprise editions)."""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self.main_window.api.remove_login_data()
        repo_config.token_remove()
        self.main_window.api.logout()
        self.main_window.api.client_reset_ssl()

        self.__authenticated = False
        self.__token = None
        self.__username = ''

        self.main_window.sig_logged_out.emit()
        telemetry.ANALYTICS.instance.event('repository-logout')
        self.update_login_status()

    def welcome_into_cloud(self) -> None:
        """Show Cloud login dialog when Navigator starts."""
        if config.CONF.get('main', 'offline_mode'):
            return

        api: cloud._CloudAPI = cloud.CloudAPI()

        if api.token:
            return

        welcome_state: int = config.CONF.get('internal', 'welcome_state', preferences.WELCOME_DELAYS.first)
        if welcome_state < preferences.WELCOME_DELAYS.first:
            return

        now = int(time.time())
        welcome_ts: int = config.CONF.get('internal', 'welcome_ts', 0)
        if welcome_ts > now:
            return

        def show_dialog(result: 'workers.TaskResult') -> None:
            if not result.result:
                return

            self.log_into_cloud()

            welcome_state: int = config.CONF.get('internal', 'welcome_state', preferences.WELCOME_DELAYS.first)
            if welcome_state < preferences.WELCOME_DELAYS.first:
                return
            config.CONF.set('internal', 'welcome_ts', now + preferences.WELCOME_DELAYS.get(welcome_state))
            if welcome_state < preferences.WELCOME_DELAYS.last:
                config.CONF.set('internal', 'welcome_state', welcome_state + 1)

        worker: 'workers.TaskWorker' = api.ping.worker()  # pylint: disable=no-member
        worker.signals.sig_succeeded.connect(show_dialog)
        worker.start()

    def setup_commercial_edition_default_channels(self, conda_rc: typing.Mapping[typing.Any, typing.Any]) -> None:
        """Setup default channels for Anaconda Professional if no CE defaults have been found"""
        commercial_edition_url: typing.Optional[str] = self.main_window.config.get('main', 'anaconda_professional_url')
        if commercial_edition_url:
            default_channels: typing.Iterable[str] = conda_rc.get('default_channels', tuple())
            if any(commercial_edition_url in channel for channel in default_channels):
                return

        configure_default_channels()

    def detect_commercial_edition_login(self, conda_rc: typing.Mapping[typing.Any, typing.Any]) -> bool:
        """Check Anaconda Professional login on the system."""
        commercial_edition_url: typing.Optional[str] = self.main_window.config.get('main', 'anaconda_professional_url')
        if any(commercial_edition_url in token_domain for token_domain in token_list()):
            self.setup_commercial_edition_default_channels(conda_rc)
            self.main_window.config.set_logged_data(commercial_edition_url, config.AnacondaBrand.COMMERCIAL_EDITION)
            return True
        return False

    def detect_team_edition_login(self) -> bool:
        """Check Anaconda Server login on the system."""
        te_detected: bool = False
        team_edition_api_url: typing.Optional[str] = self.main_window.config.get('main', 'anaconda_server_api_url')

        # Try to check if token with team edition url (specified in anaconda-navigator.ini) is present.
        if team_edition_api_url:
            for api_url, token in token_list().items():
                if team_edition_api_url in api_url and token:
                    self.main_window.config.set_logged_data(team_edition_api_url, config.AnacondaBrand.TEAM_EDITION)
                    self.main_window.config.set('main', 'anaconda_server_token', token)
                    te_detected = True
                    break

        # Anaconda Server API url wasn't specified in anaconda-navigator.ini
        # Check if user was logged in using Repo CLI.
        else:
            cli_config = get_config()
            default_site_name = cli_config.get('default_site')
            default_site_url = cli_config.get('sites', {}).get(default_site_name, {}).get('url')
            token = load_token(default_site_name)

            if default_site_name and default_site_url and token:
                parsed_url = parse.urlsplit(default_site_url)
                resulting_url = f'{parsed_url.scheme}://{parsed_url.netloc}'
                self.main_window.config.set_logged_data(resulting_url, config.AnacondaBrand.TEAM_EDITION)
                self.main_window.config.set('main', 'anaconda_server_api_url', resulting_url)
                self.main_window.config.set('main', 'anaconda_server_token', token)
                te_detected = True

        return te_detected

    def detect_anaconda_org_login(
            self,
            current_domain: typing.Union[str, None],
            current_user: typing.Union[typing.Mapping[typing.Any, typing.Any], None],
    ) -> bool:
        """Check anaconda.org login on the system."""
        anaconda_api_url: typing.Optional[str] = self.main_window.config.get('main', 'anaconda_api_url')
        if current_domain and current_domain == anaconda_api_url and current_user:
            self.main_window.config.set_logged_data(current_domain, config.AnacondaBrand.ANACONDA_ORG)
            return True
        return False

    def detect_enterprise_org_login(self, conda_rc: typing.Mapping[typing.Any, typing.Any]) -> bool:
        """Check Enterprise Edition login on the system."""
        ae4_api_url: typing.Optional[str] = self.main_window.config.get('main', 'enterprise_4_repo_api_url')
        is_ae4_alias: bool = parse.urlparse(
            ae4_api_url).netloc == parse.urlparse(conda_rc.get('channel_alias', '')).netloc

        if ae4_api_url and is_ae4_alias:
            self.main_window.config.set_logged_data(ae4_api_url, config.AnacondaBrand.ENTERPRISE_EDITION)
            return True
        return False

    def detect_new_login(self) -> typing.Union[typing.Mapping[typing.Any, typing.Any], None]:
        """ Check for new login status on the system."""
        user: typing.Optional[typing.Mapping[typing.Any, typing.Any]] = None
        conda_rc: typing.Any = self.main_window.api._conda_api.load_rc()  # pylint: disable=protected-access
        detected: bool = (
            self.detect_commercial_edition_login(conda_rc) or
            self.detect_team_edition_login() or
            self.detect_enterprise_org_login(conda_rc)
        )

        if not detected:
            user = self.main_window.api.client_user()
            domain = self.main_window.api.client_domain()
            self.detect_anaconda_org_login(domain, user)

        self.main_window.api.client_reload()
        user = user or self.main_window.api.client_user()

        return user

    def __postprocess_dialog(self, dialog: QtWidgets.QDialog, edition: str) -> None:
        """Apply changes from the login dialog."""
        if dialog.result():
            self.__authenticated = True
            self.__username = dialog.username
            self.main_window.sig_logged_in.emit()
            telemetry.ANALYTICS.instance.event('repository-login', {'edition': edition})

        self.main_window._track_tab()  # pylint: disable=protected-access
        if dialog.result():
            self.update_login_status()

    def __check_for_new_login(self) -> None:
        """
        Check for new login status periodically on the system.

        Also checks for internet connectivity and updates.
        """
        new_token: typing.Optional[str]
        new_token = self.main_window.api._client_api.load_token()  # pylint: disable=protected-access
        if new_token != self.__token:
            self.__token = new_token
            if new_token is None:
                self.log_out_from_repository()
            else:
                pass  # NOTE: How to relogin if logged from command line??

    def start_timers(self) -> None:
        """Start component timers."""
        self.__timer.start()

    def stop_timers(self) -> None:
        """Stop component timers."""
        self.__timer.stop()
