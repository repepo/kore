# -*- coding: utf-8 -*-

# pylint: disable=invalid-name,no-name-in-module,too-many-lines,unused-argument

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Main Application Window."""

from __future__ import absolute_import, division

import contextlib
import os
import typing
import psutil
from qtpy.QtCore import QPoint, QSize, Qt, QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import QApplication, QHBoxLayout, QMainWindow, QVBoxLayout, QWidget
from anaconda_navigator import __version__, __file__ as NAVIGATOR_ROOT_FILE
from anaconda_navigator.api.anaconda_api import AnacondaAPI
from anaconda_navigator.config import CHANNELS_PATH, CONF, MAC, WIN, AnacondaBrand
from anaconda_navigator.config import preferences
from anaconda_navigator.static import images
from anaconda_navigator.utils import attribution
from anaconda_navigator.utils import constants as C
from anaconda_navigator.utils import anaconda_solvers
from anaconda_navigator.utils.launch import launch
from anaconda_navigator.utils.logs import logger
from anaconda_navigator.utils.misc import set_windows_appusermodelid
from anaconda_navigator.utils import notifications
from anaconda_navigator.utils.qthelpers import create_action
from anaconda_navigator.utils.styles import BLUR_SIZE
from anaconda_navigator.utils import telemetry
from anaconda_navigator.utils import version_utils
from anaconda_navigator.utils import workers
from anaconda_navigator.widgets import ButtonBase, FrameBase, LabelBase, QSvgWidget, SpacerHorizontal
from anaconda_navigator.widgets.dialogs import MessageBoxError, MessageBoxQuestion
from anaconda_navigator.widgets.dialogs.about import AboutDialog
from anaconda_navigator.widgets.dialogs.channels import DialogChannels
from anaconda_navigator.widgets.dialogs.logger import LogViewerDialog
from anaconda_navigator.widgets.dialogs.login import TeamEditionAddChannelsPage
from anaconda_navigator.widgets.dialogs.offline import DialogOfflineMode
from anaconda_navigator.widgets.dialogs.preferences import PreferencesDialog
from anaconda_navigator.widgets.dialogs.quit import (
    ClosePackageManagerDialog, QuitApplicationDialog, QuitBusyDialog, QuitRunningAppsDialog,
)
from anaconda_navigator.widgets.dialogs.update import DialogUpdateApplication
from anaconda_navigator.widgets.tabs.community import CommunityTab
from anaconda_navigator.widgets.tabs.home import HomeTab
from anaconda_navigator.widgets.tabs.tabwidget import TabWidget
from . import account_components
from . import application_components
from . import common
from . import environment_components
from . import issue_solvers
from . import notification_components


class ComponentInitializer(typing.Protocol):  # pylint: disable=too-few-public-methods
    """Common interface for component initializers."""

    __alias__: str

    def __call__(self, parent: 'MainWindow') -> common.Component:
        """Initialize new :class:`~anaconda_navigator.main_window.common.Component` instance."""


# --- Widgets used with CSS styling
# -----------------------------------------------------------------------------

class ButtonHeaderUpdate(ButtonBase):
    """Button used in CSS styling."""


class FrameHeader(FrameBase):
    """
    Frame used in CSS styling.
    Top application header.
    """


class FrameBody(FrameBase):
    """Frame used in CSS styling."""


class LabelHeaderLogo(LabelBase):
    """Label used in CSS styling."""


class LabelHeaderUpdate(LabelBase):
    """Label used in CSS styling."""


class LabelBeta(LabelBase):
    """Label used in CSS styling."""


class LabelWarning(LabelBase):
    """Label used in CSS styling."""

    def set_offline_mode_text(self):  # pylint: disable=missing-function-docstring
        offline_text = '<i>Working in offline mode</i>'
        tooltip = DialogOfflineMode.MESSAGE_TOOL
        self.set_text(offline_text, tooltip)

    def set_text(self, text, tooltip=None):  # pylint: disable=missing-function-docstring
        self.setText(text)
        self.setToolTip(tooltip)

    def clear(self):  # pylint: disable=missing-function-docstring
        self.setText('')
        self.setToolTip('')


# --- Main widget
# -----------------------------------------------------------------------------


class MainWindowComponents(typing.Mapping[str, common.Component]):
    """
    Container for different components of the :class:`~MainWindow`.

    Created to split and group functionality of the MainWindow.
    """

    __slots__ = ('__parent', '__content')

    def __init__(self, parent: 'MainWindow') -> None:
        """Initialize new :class:`~MainWindowComponents` instance."""
        self.__parent: typing.Final[MainWindow] = parent
        self.__content: typing.Final[typing.Dict[str, common.Component]] = {}

    def push(self, component: 'ComponentInitializer') -> None:
        """Add new component to the pool."""
        key: typing.Optional[str] = getattr(component, '__alias__', None)
        if not key:
            raise ValueError('component must have a valid name')
        if key in self.__content:
            raise KeyError('component with same name is already added')
        self.__content[key] = component(parent=self.__parent)

    def for_each(self, action: typing.Callable[[common.Component], None]) -> None:
        """Apply single action to all environments."""
        component: common.Component
        for component in self.__content.values():
            action(component)

    def __getattr__(self, key: str) -> common.Component:
        """Retrieve single component as attribute."""
        try:
            return self.__content[key]
        except KeyError:
            raise AttributeError(f'{type(self).__name__} object has no attribute {key!r}') from None

    def __getitem__(self, key: str) -> common.Component:
        """Retrieve single component as item."""
        return self.__content[key]

    def __len__(self) -> int:
        """Retrieve total number of components in pool."""
        return len(self.__content)

    def __iter__(self) -> typing.Iterator[str]:
        """Iterate through component names in pool."""
        return iter(self.__content)

    # Common proxy-methods

    def setup(self, worker: typing.Any, output: typing.Any, error: str, initial: bool) -> None:
        """Perform component configuration from `conda_data`."""
        self.for_each(lambda component: component.setup(worker=worker, output=output, error=error, initial=initial))

    def update_style_sheet(self) -> None:
        """Update style sheet of the tab."""
        self.for_each(lambda component: component.update_style_sheet())

    def start_timers(self) -> None:
        """Start component timers."""
        self.for_each(lambda component: component.start_timers())

    def stop_timers(self) -> None:
        """Stop component timers."""
        self.for_each(lambda component: component.stop_timers())


class MainWindow(QMainWindow):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Main window widget."""

    sig_ready = Signal()
    sig_conda_ready = Signal()
    sig_setup_ready = Signal()
    sig_logged_in = Signal()
    sig_logged_out = Signal()

    def __init__(  # pylint: disable=too-many-arguments,too-many-branches,too-many-locals,too-many-statements
        self,
        splash=None,
        config=CONF,
        tab_home=True,
        tab_environments=True,
        tab_learning=True,
        tab_community=True,
    ):
        """Main window widget."""
        super().__init__()

        self.__components: typing.Final[MainWindowComponents] = MainWindowComponents(parent=self)
        self.components.push(notification_components.NotificationsComponent)

        anaconda_solvers.POOL.solve()

        # Variables (Global)
        self.api = AnacondaAPI()
        self.initial_setup = True
        self.setup_ready = False
        self.config = config
        self.maximized_flag = True
        self.first_run = self.config.get('main', 'first_run')
        self.application_update_version = None
        self.restart_required = None
        self._toolbar_setup_ready = False  # See issue 1142

        # Variables (Testing)
        self._dialog_about = None
        self._dialog_logs = None
        self._dialog_preferences = None
        self._dialog_update = None
        self._dialog_message_box = None
        self._dialog_quit = None
        self._dialog_quit_busy = None
        self._dialog_quit_running_apps = None
        self._dialog_offline = None

        self._dialog_channels = None
        self._dialog_environment_action = None

        # Configuration stability
        issue_solvers.CONFIGURATION_POOL.solve(
            context=issue_solvers.ConfigurationContext(
                api=self.api,
                config=self.config,
            ),
        )

        # Variables (Conda handling)
        self.busy_conda: bool = False
        self.current_prefix: str = self.config.get('main', 'default_env')

        # Fix windows displaying the right icon
        # See https://github.com/ContinuumIO/navigator/issues/1340
        if WIN:
            res = set_windows_appusermodelid()
            logger.info('appusermodelid: %s', res)

        # Widgets (Refresh timers, milliseconds)
        self._timer_offline = QTimer()  # Check for connectivity
        self._timer_offline.setInterval(4713)

        self._timer_health_check: typing.Final[QTimer] = QTimer()
        self._timer_health_check.setInterval(10000)

        # Widgets
        self.tab_home = None
        self.tab_learning = None
        self.tab_community = None
        self.splash = splash
        self.frame_header = FrameHeader(self)
        self.frame_body = FrameBody(self)
        self.label_logo = QSvgWidget(images.ANACONDA_NAVIGATOR_LOGO)
        self.label_warning = LabelWarning()
        self.button_update_available = ButtonHeaderUpdate('Upgrade Now')
        self.widget = QWidget()
        self.stack = TabWidget(self)

        # Widgets setup
        self.setWindowTitle('Anaconda Navigator')
        self.button_update_available.setVisible(False)
        self.label_logo.setFixedSize(QSize(395, 50))

        # Load custom API URL on batch installs and set it
        self.set_initial_batch_config()

        attribution.UPDATER.instance.sig_updated.connect(self.stack.add_advertisement)
        self.stack.add_advertisement()
        for link in preferences.SIDEBAR_LINKS:
            self.stack.add_link(**link._asdict())
        for social in preferences.SIDEBAR_SOCIALS:
            self.stack.add_social(**social._asdict())

        self.components.push(account_components.AccountsComponent)
        self.components.push(application_components.ApplicationsComponent)

        if tab_home:
            self.tab_home = HomeTab(parent=self)
            self.stack.addTab(self.tab_home, text='Home')

            # Signals
            self.tab_home.sig_item_selected.connect(self.select_environment)
            self.tab_home.sig_channels_requested.connect(self.show_channels)
            self.tab_home.sig_url_clicked.connect(self.open_url)
            self.tab_home.sig_launch_action_requested.connect(self.components.applications.launch_application)
            self.tab_home.sig_conda_action_requested.connect(self.components.applications.conda_application_action)

        if tab_environments:
            self.components.push(environment_components.EnvironmentsComponent)

        if tab_learning:
            self.tab_learning = CommunityTab(
                parent=self,
                tags=['webinar', 'documentation', 'video', 'training'],
                utm_medium='learning'
            )
            self.stack.addTab(self.tab_learning, text='Learning')

        if tab_community:
            self.tab_community = CommunityTab(
                parent=self,
                tags=['event', 'forum', 'social'],
                utm_medium='community'
            )
            self.stack.addTab(self.tab_community, text='Community')

        self.all_tab_widgets = [
            self.tab_home,
            self.tab_community,
            self.tab_learning,
        ]

        # Layout
        layout_header = QHBoxLayout()
        layout_header.addWidget(self.label_logo)
        layout_header.addStretch()
        layout_header.addWidget(self.label_warning)
        layout_header.addWidget(SpacerHorizontal())
        layout_header.addWidget(self.button_update_available)
        layout_header.addWidget(SpacerHorizontal())
        layout_header.addWidget(self.components.accounts.account_label_widget)
        layout_header.addWidget(SpacerHorizontal())
        layout_header.addWidget(self.components.accounts.login_button)
        self.frame_header.setLayout(layout_header)

        layout_body = QHBoxLayout()
        layout_body.addWidget(self.stack)
        layout_body.setContentsMargins(0, 0, 0, 0)
        layout_body.setSpacing(0)
        self.frame_body.setLayout(layout_body)

        layout_main = QVBoxLayout()
        layout_main.addWidget(self.frame_header)
        layout_main.addWidget(self.frame_body)
        layout_main.setContentsMargins(0, 0, 0, 0)
        layout_main.setSpacing(0)
        self.widget.setLayout(layout_main)
        self.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(self.widget)

        # Signals
        self._timer_offline.timeout.connect(self.check_internet_connectivity)
        self._timer_health_check.timeout.connect(self.api.health_check)
        self.api.sig_api_health.connect(self.show_api_health_label)
        self.button_update_available.clicked.connect(self.update_application)
        self.stack.sig_current_changed.connect(self._track_tab)
        self.stack.sig_current_changed.connect(self.fix_tab_order)
        self.stack.sig_url_clicked.connect(self.open_url)
        self.sig_setup_ready.connect(self.check_package_cache)
        self.sig_setup_ready.connect(self.check_internet_connectivity)

        # Setup
        self.config.set('main', 'last_status_is_offline', None)
        self.api.set_data_directory(CHANNELS_PATH)
        self.update_style_sheet()
        self.setMinimumWidth(1200)

        # Add featured_channels
        worker = self.api.conda_config(prefix=self.current_prefix)
        worker.sig_chain_finished.connect(self.add_featured_channels)

        worker_data = self.api.conda_data(prefix=self.current_prefix)
        worker_data.sig_chain_finished.connect(self.setup)

        attribution.UPDATER.instance.update.worker().start()  # pylint: disable=no-member
        telemetry.ANALYTICS.instance.event(
            'launch-navigator',
            user_properties=typing.cast('dict[str, str | int]', telemetry.utilities.Stats().details),
        )

    @property
    def components(self) -> MainWindowComponents:  # noqa: D401
        """Components of the :class:`~MainWindow`."""
        return self.__components

    # Helpers
    # -------------------------------------------------------------------------
    def _track_tab(self, index=None):
        """Track the active tab by index, or set `Home` when index is None."""
        if index is None:
            index = self.stack.currentIndex()

        text = self.stack.currentText()
        if text:
            text = text.lower()

        telemetry.ANALYTICS.instance.event('navigate', {'location': f'/{text}'})

    # --- Public API
    # -------------------------------------------------------------------------
    def set_initial_batch_config(self):
        """
        Set configuration settings that force conda and client config update.
        """
        def is_valid_api(url, verify):
            """Check if a given URL is a valid anaconda api endpoint."""
            output = self.api.download_is_valid_api_url(
                url,
                non_blocking=False,
                verify=verify,
            )
            return output

        verify = True

        # SSL certificate
        default_ssl_certificate = self.config.get('main', 'default_ssl_certificate')
        if default_ssl_certificate is not None:
            # Check if it is a valid path, and check if it is boolean
            if isinstance(default_ssl_certificate, bool) or os.path.isfile(default_ssl_certificate):
                self.api.client_set_ssl(default_ssl_certificate)
                # self.config.set('main', 'default_ssl_certificate', None)
                verify = default_ssl_certificate

        # API URL
        default_anaconda_api_url = self.config.get('main', 'default_anaconda_api_url')
        if default_anaconda_api_url is not None:
            if is_valid_api(default_anaconda_api_url, verify=verify):
                self.api.client_set_api_url(default_anaconda_api_url)
                # self.config.set('main', 'default_anaconda_api_url', None)

    def setup(self, worker, output, error):
        """
        Perform initial setup and configuration.

        This is the first method called after the main window has been created.
        """
        issues: notifications.NotificationCollection = issue_solvers.CONFLICT_POOL.solve(
            context=issue_solvers.ConflictContext(
                api=self.api,
                config=self.config,
                conda_info=output,
            ),
        )

        if issues.only(tags='default_env'):
            self.select_environment(prefix=self.config.get('main', 'default_env'))
            return

        self.check_internet_connectivity()

        if self.initial_setup:
            logger.info('Initial setup')

            # Get user info if it has previously logged in via anaconda client
            self.set_splash('Loading user...')

            user = self.components.accounts.detect_new_login()
            self.components.accounts.update_login_status(user)
            self.set_splash('Preparing interface...')
            self.setup_toolbars()

            self.set_splash('Loading bundled metadata...')
            self.api.load_bundled_metadata()

            self.post_setup(conda_data=output)

        else:
            logger.info('Normal setup')

            # Reset home and environment tab
            if self.tab_home:
                self.tab_home.setup(output)

            # Check for updates
            packages = output['packages']
            info = output.get('processed_info', {})
            is_root_writable = info.get('root_writable', False)
            self.check_for_updates(packages=packages, is_root_writable=is_root_writable)

            self.fix_tab_order()

        self.components.setup(worker=worker, output=output, error=error, initial=self.initial_setup)
        self.initial_setup = False

    def post_setup(self, conda_data):
        """Setup tab widgets."""
        self.setup_ready = True
        self.config.set('main', 'first_run', False)

        if self.tab_home:
            self.set_busy_status(conda=True)
            self.set_splash('Loading applications...')
            self.tab_home.setup(conda_data)
        if 'environments' in self.components:
            self.set_busy_status(conda=True)
            self.set_splash('Loading environments...')
            self.components.environments.tab.setup(conda_data)
        if self.tab_community:
            self.set_splash('Loading content...')
            self.tab_community.setup()
        if self.tab_learning:
            self.set_splash('Loading content...')
            self.tab_learning.setup()
        self.update_style_sheet()

        geo = self.config.get('main', 'geo', None)
        if geo:
            # In case Navigator is installed in an env that has a different
            # python version than root (root == py3, env == py2 or viceversa)
            try:
                self.restoreGeometry(geo)
                self.show()
            except BaseException:  # pylint: disable=broad-except
                self.showMaximized()
        else:
            self.showMaximized()
        self.post_visible_setup(conda_data)

    def setup_toolbars(self):
        """Setup toolbar menus and actions."""
        # See issue #1142
        if self._toolbar_setup_ready:
            return

        menubar = self.menuBar()

        file_menu = menubar.addMenu('&File')
        file_menu.addAction(create_action(self, '&Preferences', triggered=self.show_preferences, shortcut='Ctrl+P'))
        file_menu.addAction(create_action(self, '&Restart', triggered=self.restart, shortcut='Ctrl+R'))
        file_menu.addAction(create_action(self, '&Quit', triggered=self.close, shortcut='Ctrl+Q'))

        helpmenu = menubar.addMenu('&Help')
        helpmenu.addAction(create_action(self, '&Online Documentation', triggered=self.open_online_documentation))
        helpmenu.addAction(create_action(self, '&Logs viewer', triggered=self.show_log_viewer, shortcut='F6'))
        helpmenu.addSeparator()
        helpmenu.addAction(create_action(self, '&About', triggered=self.show_about))
        self._toolbar_setup_ready = True

    def post_visible_setup(self, conda_data):
        """Setup after show method has been applied."""
        if self.splash:
            self.splash.hide()

        self.config.set('main', 'first_run', False)

        # Start the tracker only after post_visible_setup
        self._track_tab(0)  # Start tracking home

        packages = conda_data.get('packages')
        info = conda_data.get('processed_info', {})
        is_root_writable = info.get('root_writable', False)
        self.check_for_updates(packages=packages, is_root_writable=is_root_writable)

        # Fix tab order
        self.fix_tab_order()
        buttons = self.stack.tabbar.buttons
        if buttons:
            buttons[0].setFocus()

        worker = self.api.conda_config_and_sources(prefix=self.current_prefix)
        worker.sig_chain_finished.connect(self.check_outdated_channels)

        self.check_internet_connectivity()
        self.sig_setup_ready.emit()

        self.components.accounts.welcome_into_cloud()

    def set_widgets_enabled(self, value):
        """Set the widgets enabled/disabled status for subwidgets and tabs."""
        if self.tab_home:
            self.tab_home.set_widgets_enabled(value)
        if 'environments' in self.components:
            self.components.environments.tab.set_widgets_enabled(value)

    def update_style_sheet(self):
        """Update custom CSS style sheet."""
        for tab in self.all_tab_widgets:
            if tab:
                tab.update_style_sheet()
        self.components.update_style_sheet()

    # --- Update Navigator
    # -------------------------------------------------------------------------
    def check_for_updates(self, packages=None, version=None, is_root_writable=False):
        """Check for application updates."""
        # Check if there is an update for navigator!
        navi_version = version or __version__

        self.button_update_available.setEnabled(False)
        self.button_update_available.setVisible(False)

        if packages:
            package_data = packages.get('anaconda-navigator')
            if package_data:
                versions = package_data.get('versions')
                if versions and (version_utils.compare(versions[-1], navi_version) > 0):
                    self.application_update_version = versions[-1]
                    self.button_update_available.setEnabled(True)
                    self.button_update_available.setVisible(True)
                    if not self.config.get('main', 'hide_update_dialog'):
                        self.update_application(
                            center_dialog=True,
                            is_root_writable=is_root_writable,
                        )

    def update_application(self, center_dialog=False, is_root_writable=False):
        """Update application to latest available version."""
        version = self.application_update_version
        qa_testing = version == '1000.0.0'

        if version:
            dlg = DialogUpdateApplication(
                version=version,
                startup=center_dialog,
                qa_testing=qa_testing,
                is_root_writable=is_root_writable,
            )
            # Only display one dialog at a time
            if self._dialog_update is None:

                self._dialog_update = dlg
                if not center_dialog:
                    height = self.button_update_available.height()
                    width = self.button_update_available.width()
                    point = self.button_update_available.mapToGlobal(QPoint(-dlg.WIDTH + width, height))
                    dlg.move(point)

                telemetry.ANALYTICS.instance.event('navigate', {'location': '/update'})

                if dlg.exec_():
                    telemetry.ANALYTICS.instance.event('update-navigator', {'from': __version__, 'to': version})
                    # Returns a pid or None if failed
                    pid = self.open_updater(version, is_root_writable=is_root_writable)
                    if pid is not None:
                        self.close()

            self._dialog_update = None
            self._track_tab()

    def open_updater(self, version, is_root_writable=False):
        """Open the Anaconda Navigator Updater"""
        leave_path_alone = True
        root_prefix = self.api.ROOT_PREFIX
        prefix = os.environ.get('CONDA_PREFIX', root_prefix)
        command = f'navigator-updater --latest-version {version} --prefix {prefix}'

        as_admin = WIN and not is_root_writable

        return launch(
            root_prefix=root_prefix,
            prefix=prefix,
            command=command,
            package_name='anaconda-navigator-updater',
            leave_path_alone=leave_path_alone,
            non_conda=True,
            as_admin=as_admin,
        )

    # --- Url handling
    # -------------------------------------------------------------------------
    # NOTE: Route ALL url handling to this method? or Make a global func?
    def open_url(self, url, category=None, action=None):
        """Open url and track event."""
        QDesktopServices.openUrl(QUrl(url))
        telemetry.ANALYTICS.instance.event('redirect', {'url': str(url)})

    # --- Client (Login)
    # -------------------------------------------------------------------------
    @property
    def conda_url(self):
        """Return the conda url based on the api info from config url."""
        api_info = self.api.download_get_api_info()
        url = api_info.get('conda_url', 'https://conda.anaconda.org')
        url = url[:-1] if url[-1] == '/' else url
        return url

    @property
    def api_url(self) -> str:
        """Return the api url from anaconda client config."""
        return self.api.client_get_api_url()

    # --- Dialogs
    # -------------------------------------------------------------------------
    def show_preferences(self):
        """Display the preferences dialog and apply the needed actions."""
        telemetry.ANALYTICS.instance.event('navigate', {'location': '/preferences'})
        self._dialog_preferences = PreferencesDialog(
            parent=self,
            config=self.config,
            environments=self.components.environments.environments,
        )

        # If the api url was changed by the user, a logout is triggered
        self._dialog_preferences.sig_urls_updated.connect(
            lambda au, cu: self.components.accounts.log_out_from_repository(),
        )
        self._dialog_preferences.exec_()
        self._dialog_preferences = None
        self._track_tab()

    def show_about(self):
        """Display the `About` dialog with information on the project."""
        self._dialog_about = AboutDialog(self)
        telemetry.ANALYTICS.instance.event('navigate', {'location': '/about'})
        self._dialog_about.sig_url_clicked.connect(self.open_url)
        self._dialog_about.exec_()
        self._dialog_about = None
        self._track_tab()

    def open_online_documentation(self) -> None:
        """Open documentation page in web browser."""
        self.open_url('https://docs.anaconda.com/free/navigator?utm_source=anaconda_navigator&utm_medium=nav-help')

    def show_log_viewer(self):
        """Display the logs viewer to the user."""
        self._dialog_logs = LogViewerDialog()
        telemetry.ANALYTICS.instance.event('navigate', {'location': '/logs'})
        self._dialog_logs.exec_()
        self._dialog_logs = None
        self._track_tab()

    def show_error_message(self, error_name, error_text):
        """Display application error message."""
        self.set_busy_status(conda=True)
        if 'UnsatisfiableSpecifications' in error_name:
            report = False
        else:
            report = True

        if report:
            title = 'Conda process error'
            text = 'The following errors occurred:'
            error_msg = error_text
        else:
            title = 'Unsatisfiable package specifications:'
            text = 'The following specifications were found to be in conflict:'
            package_errors = [e.strip() for e in error_text.split('\n') if '-' in e]
            error_msg = '\n'.join(package_errors)
            error_msg += '\n\n\nA package you tried to install conflicts with another.'

        # Check if offline mode and provide a custom message
        if self.api.is_offline():
            report = False
            if 'PackagesNotFoundError' in error_name:
                title = 'Package not available in offline mode'
                error_msg = (
                    'Some of the functionality of Anaconda Navigator will be '  # pylint: disable=implicit-str-concat
                    'limited in <b>offline mode</b>.<br><br>Installation and upgrade of packages will be subject to '
                    'the packages currently available on your package cache.'
                )

        telemetry.ANALYTICS.instance.event('navigate', {'location': '/conda_error'})
        dlg = MessageBoxError(
            text=text,
            error=error_msg,
            title=title,
            report=False,  # Disable reporting on github
            learn_more='http://conda.pydata.org/docs/troubleshooting.html#unsatisfiable'
        )
        self._dialog_message_box = dlg
        dlg.setMinimumWidth(400)
        dlg.exec_()
        self.set_busy_status(conda=False)
        self.update_status()
        self._track_tab()

    def show_offline_mode_dialog(self):
        """Show offline mode dialog"""
        if self._dialog_offline is not None:
            # Dialog currently open
            return

        show_dialog = not self.config.get('main', 'hide_offline_dialog')
        first_time_offline = self.config.get('main', 'first_time_offline')

        if show_dialog or first_time_offline:
            self._dialog_offline = DialogOfflineMode(parent=self)
            telemetry.ANALYTICS.instance.event('navigate', {'location': '/offline'})

            if self._dialog_offline.exec_():
                pass

            self.config.set('main', 'first_time_offline', False)
            self._dialog_offline = None
            self._track_tab()

    # --- Conda (Dialogs)
    # -------------------------------------------------------------------------
    def add_default_channels(self, conda_data):
        """Add defaults to user condarc if it does not exist."""
        config_sources = conda_data.get('config_sources', {})
        user_rc = self.api._conda_api.user_rc_path  # pylint: disable=protected-access
        user_config = config_sources.get(user_rc, {})
        user_channels = user_config.get('channels', [])
        if len(user_channels) == 0 or not os.path.isfile(user_rc):
            worker = self.api.conda_config_add('channels', 'defaults')
            worker.communicate()
        self.api.client_get_ssl()

    def check_outdated_channels(self, worker, conda_data, error):
        """Check if the user has binstar channels as part of the config."""
        self.set_busy_status(conda=True)
        config = conda_data.get('config', {})
        current_channels = config.get('channels', [])
        self.add_default_channels(conda_data)
        outdated_channels = []
        if 'anaconda.org' in self.api_url:  # pylint: disable=unsupported-membership-test
            for channel in current_channels:
                if 'binstar.org' in channel:
                    outdated_channels.append(channel)

        if outdated_channels:
            chs = [f'- <b>{ch}</b><br>' for ch in outdated_channels]
            chs = ''.join(chs)
            text = (
                f'We detected some outdated channels making use of <b>binstar.org</b><br><br>'
                f'{chs}<br><br>'
                f'Do you want to update them?'
            )

            telemetry.ANALYTICS.instance.event('navigate', {'location': '/outdated_channels'})
            self._dialog_message_box = MessageBoxQuestion(text=text, title='Outdated channels detected')
            self._dialog_message_box.setMinimumWidth(300)

            if self._dialog_message_box.exec_():
                for channel in outdated_channels:
                    new_channel = channel.replace('binstar.org', 'anaconda.org')
                    worker = self.api.conda_config_add('channels', new_channel)
                    worker.communicate()
                    worker = self.api.conda_config_remove('channels', channel)
                    worker.communicate()
            self._dialog_message_box = None
            self._track_tab()
        self.set_busy_status(conda=False)

    def show_channels(self, button=None, sender=None):
        """Show the conda channels configuration dialog."""
        brand = CONF.get('main', 'logged_brand')
        if brand == AnacondaBrand.TEAM_EDITION:
            telemetry.ANALYTICS.instance.event('navigate', {'location': '/server_channels'})
            if TeamEditionAddChannelsPage(btn_cancel_msg='Cancel').exec_():
                self.update_index(self)
            self._track_tab()
            return

        def _accept_channels_dialog(button):
            button.setEnabled(True)
            button.setFocus()
            button.toggle()
            self._dialog_channels = None  # for testing

        if sender == C.TAB_HOME:
            pass
        elif sender == C.TAB_ENVIRONMENT:
            pass

        telemetry.ANALYTICS.instance.event('navigate', {'location': '/channels'})
        dlg = DialogChannels(parent=self)
        self._dialog_channels = dlg  # For testing
        dlg.update_style_sheet()
        worker = self.api.conda_config_sources(prefix=self.current_prefix)
        worker.sig_chain_finished.connect(dlg.setup)
        dlg.sig_channels_updated.connect(self.update_channels)

        if button:
            button.setDisabled(True)
            dlg.rejected.connect(lambda: button.setEnabled(True))
            dlg.rejected.connect(button.toggle)
            dlg.rejected.connect(button.setFocus)
            dlg.accepted.connect(lambda v=None: _accept_channels_dialog(button))

            geo_tl = button.geometry().topLeft()
            tl = button.parentWidget().mapToGlobal(geo_tl)
            x = tl.x() - BLUR_SIZE
            y = tl.y() + button.height() - BLUR_SIZE

            # Ensure channels dialog stays visible
            # See https://github.com/ContinuumIO/navigator/issues/1254
            x_right_dialog = x + dlg.WIDTH
            x_right_screen = QApplication.desktop().availableGeometry().width()
            if x_right_dialog > x_right_screen:
                x = x - (x_right_dialog - x_right_screen)
            elif x < 0:
                x = 0

            dlg.move(x, y)

        if dlg.exec_():
            pass

        dlg.button_add.setFocus()
        self._track_tab()

    def show_cancel_process(self):
        """Allow user to cancel an ongoing process."""
        if self.is_busy():
            dlg = ClosePackageManagerDialog(parent=self)
            self._dialog_quit_busy = dlg

            if self._dialog_quit_busy.exec_():
                self.update_status(action='Process cancelled', message=None)
                self.api.conda_terminate()
                self.api.download_terminate()
                self.api.conda_clear_lock()

            self.current_prefix = self.api.ROOT_PREFIX
            self.set_busy_status(conda=False)
            self.select_environment(prefix=self.api.ROOT_PREFIX)

    # --- Conda
    # -------------------------------------------------------------------------

    def select_environment(self, name=None, prefix=None, sender=None):
        """Select the active conda environment of the application."""
        self.set_busy_status(conda=True)
        self.current_prefix = prefix
        if 'environments' in self.components:
            self.components.environments.tab.set_loading(prefix=prefix, value=True)
        msg = f'Loading packages of <b>{prefix}</b>...'
        self.update_status(action=msg, value=0, max_value=0)
        self.set_widgets_enabled(False)
        worker = self.api.conda_data(prefix=self.current_prefix)
        worker.sig_chain_finished.connect(self.setup)

    def check_package_cache(self) -> None:
        """Check if package cache is obsolete and should be updated."""
        timestamp: float = self.api._conda_api.get_repodata_modification_time()  # pylint: disable=protected-access
        if timestamp < max(os.path.getctime(NAVIGATOR_ROOT_FILE), os.path.getmtime(NAVIGATOR_ROOT_FILE)):
            self.update_index(self)

    def update_index(self, sender):
        """Update conda repodata index."""
        self.set_busy_status(conda=True)
        self.update_status('Updating package index and metadata...', '', value=0, max_value=0)
        worker = self.api.update_index_and_metadata(prefix=self.current_prefix)
        worker.sig_chain_finished.connect(self._conda_output_ready)

    def update_channels(self, sources_added, sources_removed):
        """Save updated channels to the conda config."""
        self.update_status(
            action='Updating channel configuration...',
            value=0,
            max_value=0,
        )

        for (source, channel) in sources_added:
            worker = self.api.conda_config_add('channels', channel, file=source)
            worker.communicate()
        for (source, channel) in sources_removed:
            worker = self.api.conda_config_remove('channels', channel, file=source)
            worker.communicate()

        worker = self.api.update_index_and_metadata(prefix=self.current_prefix)
        worker.sig_chain_finished.connect(self._conda_output_ready)

    def add_featured_channels(self, worker, conda_config, error):
        """Automatically add featured channels on startup if not found."""
        self.set_splash('Adding featured channels...')
        add_channels = self.config.get('main', 'add_default_channels')
        current_channels = conda_config.get('channels', [])
        if 'anaconda.org' in self.api_url and add_channels:
            for channel in preferences.FEATURED_CHANNELS:
                if channel not in current_channels:
                    worker = self.api.conda_config_add('channels', channel)
                    worker.communicate()
            self.config.set('main', 'add_default_channels', False)

    def _conda_partial_output_ready(self, worker, output, error):
        """Callback."""
        self.set_busy_status(conda=True)

        action_msg = worker.action_msg

        # Get errors and data from output if it exists
        if not isinstance(output, dict):
            output = {}

        # name = output.get('name')  # Linking step gone?
        fetch = output.get('fetch')  # Fetching step
        value = output.get('progress', 0)
        max_value = output.get('maxval', 0)

        if fetch:
            message = f'Fetching <b>{fetch}</b>...'
            self.update_status(
                action=action_msg,
                message=message,
                value=int(value * 100),
                max_value=int(max_value * 100),
            )

    def _conda_output_ready(self, worker, output, error):
        """Callback for handling action finished."""
        self.set_busy_status(conda=False)

        action = worker.action
        if not isinstance(output, dict):
            output = {}

        error_text = output.get('error', '')
        exception_type = output.get('exception_type', '')
        exception_name = output.get('exception_name', '')
        # import from yaml provides empty dic, hence the True
        success = output.get('success', True)

        # Check if environment was created. Conda env does not have --json output, so we check if folder was created
        if action == C.ACTION_IMPORT:
            success = os.path.isdir(worker.prefix)

        is_error = error_text or exception_type or exception_name

        # Set the current prefix to the prefix stablihsed by worker
        old_prefix = getattr(worker, 'old_prefix', None)
        prefix = getattr(worker, 'prefix', old_prefix)

        # Set as current environment only if a valid environment
        if prefix and self.api.conda_environment_exists(prefix=prefix):
            self.current_prefix = prefix
        elif old_prefix and self.api.conda_environment_exists(prefix=old_prefix):
            # If there is an error when installing an application in a new
            # environment due to conflicts, restore the previous prefix
            self.current_prefix = old_prefix
        else:
            self.current_prefix = self.api.ROOT_PREFIX

        if is_error or error or not success:
            logger.error(error_text)
            self.show_error_message(exception_name, error_text)
            self.select_environment(prefix=self.current_prefix)
        else:
            if action == C.ACTION_REMOVE_ENV:
                self.select_environment(prefix=self.api.ROOT_PREFIX)
            else:
                self.select_environment(prefix=self.current_prefix)

    def check_internet_connectivity(self):
        """Check if there is internet available."""
        last_status_is_offline = self.config.get('main', 'last_status_is_offline')
        is_offline = self.api.is_offline()

        if is_offline != last_status_is_offline and self.setup_ready:
            self.config.set('main', 'last_status_is_offline', is_offline)

            if is_offline:
                # Disable login/logout button
                self.components.accounts.login_button.setDisabled(True)

                # Include label to indicate mode
                self.label_warning.set_offline_mode_text()
                self.show_offline_mode_dialog()
            else:
                # Restore buttons and text
                self.components.accounts.login_button.setEnabled(True)
                self.label_warning.clear()

    def show_api_health_label(self, healthy: bool) -> None:
        """Show error icon if TE api is not available"""
        if not healthy:
            self.tab_home.te_alert.show_error()
            self.components.environments.tab.widget.te_alert.show_error()
            self.components.accounts.show_error_icon(
                tooltip='Some things in the Home and Environments tabs may not work because '
                        'Anaconda Server can’t be reached. We will attempt to reconnect you periodically.'
            )
            return

        self.components.accounts.hide_error_icon()

        if self.config.get_logged_data()[0] == AnacondaBrand.TEAM_EDITION:
            self.tab_home.te_alert.show_info()
            self.components.environments.tab.widget.te_alert.show_info()
        else:
            self.tab_home.te_alert.hide_all()
            self.components.environments.tab.widget.te_alert.hide_all()

    def stop_timers(self):
        """Stop all refreshing timers."""
        self._timer_offline.stop()
        self.components.stop_timers()

    def start_timers(self):
        """Start all refreshing timers."""
        self._timer_offline.start()
        self._timer_health_check.start()
        self.components.start_timers()

    def fix_tab_order(self):
        """Fix tab order of UI widgets."""
        current_widget = self.stack.currentWidget()
        if current_widget is not None:
            ordered_widgets = [
                self.button_update_available,
                self.components.accounts.login_button,
            ]
            ordered_widgets += self.stack.tabbar.buttons
            next_widdget = self.stack.tabbar.links[0]
            ordered_widgets += current_widget.ordered_widgets(next_widdget)
            ordered_widgets += self.stack.tabbar.links
            ordered_widgets += self.stack.tabbar.links_social
            ordered_widgets += [self.button_update_available]

            for index in range(len(ordered_widgets) - 1):
                self.setTabOrder(ordered_widgets[index], ordered_widgets[index + 1])

    def restart(self):
        """Restart application."""
        root_prefix = self.api.ROOT_PREFIX
        prefix = os.environ.get('CONDA_PREFIX', root_prefix)
        leave_path_alone = True
        command = 'anaconda-navigator'
        self.restart_required = False
        if self.closing():
            launch(
                root_prefix=root_prefix,
                prefix=prefix,
                command=command,
                package_name='anaconda-navigator-restart',
                leave_path_alone=leave_path_alone,
            )
            self.restart_required = True
            self.close()

    def set_splash(self, message):
        """Set splash dialog message."""
        if self.splash:
            self.splash.show_message(message)
        QApplication.processEvents()

    def toggle_fullscreen(self):
        """Toggle fullscreen status."""
        if self.isFullScreen():
            if self.maximized_flag:
                self.showMaximized()
            else:
                self.showNormal()
        else:
            self.maximized_flag = self.isMaximized()
            self.showFullScreen()

    def set_busy_status(self, conda=None):
        """
        Update the busy status of conda and the application.

        Conda status is defined by actions taken on Home/Environments tab.

        The value will only update if True or False, if None, the current value
        set will remain.
        """
        if conda is not None and isinstance(conda, bool):
            self.busy_conda = conda
            if self.busy_conda:
                self.stop_timers()
            else:
                self.start_timers()

        if not self.busy_conda:
            self.sig_conda_ready.emit()

        if not any([self.busy_conda]):
            self.sig_ready.emit()

    def is_busy(self):
        """Return if the application is currently busy."""
        return self.busy_conda

    def update_status(self, action=None, message=None, value=None, max_value=None):
        """Update status bar."""
        if self.tab_home:
            self.tab_home.update_status(action=action, message=message, value=value, max_value=max_value)
        if 'environments' in self.components:
            self.components.environments.tab.update_status(
                action=action,
                message=message,
                value=value,
                max_value=max_value,
            )

    def closing(self):  # pylint: disable=too-many-branches
        """Closing helper method to reuse on close event and restart."""
        close = True

        if os.environ.get('TEST_CI') is not None:
            return close

        if (not self.is_busy()) and (not self.config.get('main', 'hide_running_apps_dialog')):
            self.components.applications.update_running_processes()
            running_processes = self.components.applications.running_processes
            if running_processes:
                telemetry.ANALYTICS.instance.event('navigate', {'location': '/quit_running'})
                dialog = QuitRunningAppsDialog(parent=self, running_processes=running_processes)
                if dialog.exec_():
                    close_apps = self.config.get('main', 'running_apps_to_close')
                    for running_process in running_processes:
                        if running_process.package not in close_apps:
                            continue

                        process = psutil.Process(running_process.pid)
                        for child in process.children(recursive=True):
                            with contextlib.suppress(BaseException):
                                child.kill()

                        with contextlib.suppress(BaseException):
                            process.kill()
                else:
                    close = False
                self._track_tab()

        if close:
            if self.is_busy():
                telemetry.ANALYTICS.instance.event('navigate', {'location': '/quit_busy'})
                self._dialog_quit_busy = QuitBusyDialog(parent=self)
                if not self._dialog_quit_busy.exec_():
                    close = False
                self._dialog_quit_busy = None
                self._track_tab()
            else:
                show_dialog = not self.config.get('main', 'hide_quit_dialog')
                if show_dialog:
                    telemetry.ANALYTICS.instance.event('navigate', {'location': '/quit'})
                    self._dialog_quit = QuitApplicationDialog(parent=self)
                    if not self._dialog_quit.exec_():
                        close = False
                        self._track_tab()
                    self._dialog_quit = None

        return close

    # --- Qt methods
    # -------------------------------------------------------------------------
    def closeEvent(self, event):
        """Catch close event."""
        # If a restart was required don't ask self.closing again
        if (not self.restart_required) and (not self.closing()):
            event.ignore()
            return

        try:
            self.config.set('main', 'geo', bytes(self.saveGeometry()))
        except Exception as e:  # pylint: disable=broad-except
            logger.error(e)

        # hide main window showing that all primary activity is done
        self.hide()

        # wait for background tasks to finish
        workers.MANAGER.instance.cancel_all()
        workers.THREAD_POOL.reset()

    def keyPressEvent(self, event):
        """Override Qt method."""
        key = event.key()
        modifiers = event.modifiers()
        if key == Qt.Key_F5:
            self.update_style_sheet()
        elif key == Qt.Key_F11 and not MAC:
            self.toggle_fullscreen()
        elif key == Qt.Key_F and modifiers & Qt.ControlModifier and MAC:
            self.toggle_fullscreen()

        super().keyPressEvent(event)
