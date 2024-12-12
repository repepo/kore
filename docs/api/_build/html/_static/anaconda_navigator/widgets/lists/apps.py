# -*- coding: utf-8 -*-

# pylint: disable=no-name-in-module

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""
Widgets to list applications available to launch from the Home tab.

This widget does not perform the actual conda actions or command launch, but it
emits signals that should be connected to the parents and final controller on
the main window.
"""

from __future__ import absolute_import, division, print_function

import typing

from qtpy.QtCore import QPoint, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QHBoxLayout, QListWidget, QMenu, QVBoxLayout

from anaconda_navigator.api import external_apps
from anaconda_navigator.api.anaconda_api import AnacondaAPI
from anaconda_navigator.static.images import ANACONDA_ICON_256_PATH
from anaconda_navigator.utils import constants as C
from anaconda_navigator.utils.constants import AppType
from anaconda_navigator.utils.py3compat import to_text_string
from anaconda_navigator.utils.qthelpers import add_actions, create_action, update_pointer
from anaconda_navigator.utils.styles import SASS_VARIABLES
from anaconda_navigator.utils import telemetry
from anaconda_navigator.widgets import ButtonLabel, ButtonNormal, FrameBase, LabelBase
from anaconda_navigator.widgets.lists import ListWidgetBase, ListWidgetItemBase
from anaconda_navigator.widgets.spinner import NavigatorSpinner


# --- Widgets used in CSS styling
# -----------------------------------------------------------------------------

class ButtonApplicationInstall(ButtonNormal):
    """Button used in CSS styling."""


class ButtonApplicationLaunch(ButtonNormal):
    """Button used in CSS styling."""


class ButtonApplicationOptions(ButtonNormal):
    """Button used in CSS styling."""


class ButtonApplicationUpdate(ButtonNormal):
    """Button used in CSS styling."""


class LabelApplicationIcon(LabelBase):
    """Label used in CSS styling."""


class LabelApplicationName(LabelBase):
    """Label used in CSS styling."""


class LabelApplicationVersion(LabelBase):
    """Label used in CSS styling."""


class LabelApplicationDescription(LabelBase):
    """Label used in CSS styling."""


class FrameApplicationSpinner(FrameBase):
    """Label used in CSS styling."""


class ButtonApplicationVersion(ButtonLabel):  # pylint: disable=too-few-public-methods
    """Button used in CSS styling."""


class WidgetApplication(FrameBase):
    """Widget used in CSS styling."""

    # application_name, command, leave_path_alone, prefix, sender, non_conda, app_type
    sig_launch_action_requested = Signal(object, object, object, bool, object, object, object, object)

    # action, application_name, version, sender, non_conda, app_type
    sig_conda_action_requested = Signal(object, object, object, object, object, object)

    sig_url_clicked = Signal(object)


# --- Main Widgets
# -----------------------------------------------------------------------------
class ListWidgetApplication(ListWidgetBase):
    """Widget that holds the whole list of applications to launch."""

    # application_name, command, extra_arguments, leave_path_alone, prefix, sender, non_conda, app_type
    sig_launch_action_requested = Signal(object, object, object, bool, object, object, object, object)

    # action, application_name, version, sender, non_conda, app_type
    sig_conda_action_requested = Signal(object, object, object, object, object, object)

    sig_url_clicked = Signal(object)

    def __init__(self, *args, **kwargs):
        """Widget that holds the whole list of applications to launch."""
        super().__init__(*args, **kwargs)
        self.setGridSize(ListItemApplication.widget_size())
        self.setWrapping(True)
        self.setViewMode(QListWidget.IconMode)
        self.setLayoutMode(ListWidgetApplication.Batched)
        self.setFocusPolicy(Qt.NoFocus)

    def ordered_widgets(self):
        """Return a list of the ordered widgets."""
        ordered_widgets = []
        for item in self.items():
            ordered_widgets += item.ordered_widgets()
        return ordered_widgets

    def setup_item(self, item):
        """Override base method."""
        item.widget.sig_conda_action_requested.connect(self.sig_conda_action_requested)
        item.widget.sig_launch_action_requested.connect(self.sig_launch_action_requested)
        item.widget.sig_url_clicked.connect(self.sig_url_clicked)


class ListItemApplication(ListWidgetItemBase):  # pylint: disable=too-many-instance-attributes
    """Item with custom widget for the applications list."""

    ICON_HEIGHT = 64  # pylint: disable=invalid-name
    ICON_WIDTH = 192  # pylint: disable=invalid-name

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
        self,
        name: str,
        display_name: typing.Optional[str] = None,
        description: typing.Optional[str] = None,
        command: typing.Optional[str] = None,
        extra_arguments: typing.Sequence[typing.Any] = (),
        version: typing.Optional[str] = None,
        versions: typing.Optional[typing.Sequence[str]] = None,
        image_path: typing.Optional[str] = None,
        prefix: typing.Optional[str] = None,
        non_conda: bool = False,
        installed: bool = False,
        summary: typing.Optional[str] = None,
        app_type: AppType = AppType.CONDA,
        rank: int = 0,
    ) -> None:
        """Item with custom widget for the applications list."""
        super().__init__()

        self.api = AnacondaAPI()
        self.prefix = prefix
        self.name: str = name
        self.display_name = display_name if display_name else name
        self.url = ''
        self.expired = False
        self.description = description if description else summary
        self.command = command
        self.extra_arguments = extra_arguments
        self.version = version
        self.versions = versions
        self.image_path = image_path if image_path else ANACONDA_ICON_256_PATH
        self.timeout = 2000
        self.non_conda = non_conda
        self.installed = installed
        self.app_type = app_type
        self.rank = rank

        # Widgets
        self.button_install = ButtonApplicationInstall('Install')  # or Try!
        self.button_launch = ButtonApplicationLaunch('Launch')
        self.button_options = ButtonApplicationOptions()
        self.label_icon = LabelApplicationIcon()
        self.label_name = LabelApplicationName(self.display_name)
        self.label_description = LabelApplicationDescription(self.description)
        self.button_version = ButtonApplicationVersion(to_text_string(self.version))
        self.menu_options = QMenu('Application options')
        self.menu_versions = QMenu('Install specific version')
        self.pixmap = QPixmap(self.image_path)
        self.timer = QTimer()
        self.widget = WidgetApplication()
        self.frame_spinner = FrameApplicationSpinner()
        self.spinner = NavigatorSpinner(self.widget, total_width=16)
        lay = QHBoxLayout()
        lay.addWidget(self.spinner)
        self.frame_spinner.setLayout(lay)

        # Scale icon
        icon_width: int = self.pixmap.width() * self.ICON_HEIGHT
        icon_height: int = self.pixmap.height() * self.ICON_WIDTH
        if icon_width >= icon_height:
            icon_width = self.ICON_WIDTH
            icon_height //= self.pixmap.width()
        else:
            icon_width //= self.pixmap.height()
            icon_height = self.ICON_HEIGHT

        # Widget setup
        self.button_version.setFocusPolicy(Qt.NoFocus)
        self.button_version.setEnabled(True)
        self.label_description.setAlignment(Qt.AlignCenter)
        self.timer.setInterval(self.timeout)
        self.timer.setSingleShot(True)
        self.label_icon.setPixmap(self.pixmap)
        self.label_icon.setScaledContents(True)  # important on High DPI!
        self.label_icon.setFixedSize(icon_width, icon_height)
        self.label_icon.setAlignment(Qt.AlignCenter)
        if self.image_path == ANACONDA_ICON_256_PATH:
            if not installed:
                self.label_icon.setToolTip('Application icon will show when installed')
            else:
                self.label_icon.setToolTip('Application icon is missing from the package recipe')

        self.label_name.setAlignment(Qt.AlignCenter)
        self.label_name.setWordWrap(True)
        self.label_name.setFixedWidth(200)
        self.label_description.setWordWrap(True)
        self.label_description.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        self.frame_spinner.setVisible(False)

        # Layouts
        layout_spinner = QHBoxLayout()
        layout_spinner.addWidget(self.button_version, 0, Qt.AlignCenter)
        layout_spinner.addWidget(self.frame_spinner, 0, Qt.AlignCenter)

        layout_main = QVBoxLayout()
        layout_main.addWidget(self.button_options, 0, Qt.AlignRight)
        layout_main.addWidget(self.label_icon, 0, Qt.AlignCenter)
        layout_main.addWidget(self.label_name, 0, Qt.AlignCenter)
        layout_main.addLayout(layout_spinner)
        layout_main.addWidget(self.label_description, 0, Qt.AlignCenter)
        layout_main.addWidget(self.button_launch, 0, Qt.AlignCenter)
        layout_main.addWidget(self.button_install, 0, Qt.AlignCenter)

        self.widget.setLayout(layout_main)
        self.setSizeHint(self.widget_size())
        # This might help with visual quirks on the home screen
        self.widget.setMinimumSize(self.widget_size())

        # Signals
        self.button_install.clicked.connect(self.install_application)
        self.button_launch.clicked.connect(self.launch_application)
        self.button_options.clicked.connect(self.actions_menu_requested)
        self.timer.timeout.connect(self._application_launched)

        # Setup
        self.update_status()

    # --- Callbacks
    # -------------------------------------------------------------------------
    def _application_launched(self):
        self.button_launch.setDisabled(False)
        update_pointer()

    # --- Helpers
    # -------------------------------------------------------------------------
    def update_style_sheet(self):
        """Update custom CSS stylesheet."""

    def ordered_widgets(self):
        """Return a list of the ordered widgets."""
        return [self.button_install, self.button_launch, self.button_options]

    @staticmethod
    def widget_size():
        """Return the size defined in the SASS file."""
        return QSize(SASS_VARIABLES.WIDGET_APPLICATION_TOTAL_WIDTH, SASS_VARIABLES.WIDGET_APPLICATION_TOTAL_HEIGHT)

    def launch_url(self):
        """Launch signal for url click."""
        self.widget.sig_url_clicked.emit(self.url)

    def actions_menu_requested(self):
        """Create and display menu for the currently selected application."""
        self.menu_options.clear()
        self.menu_versions.clear()

        # Add versions menu
        versions = self.versions if self.versions else []
        version_actions = []
        for version in reversed(versions):
            action = create_action(
                self.widget,
                version,
                triggered=lambda value, version=version: self.install_application(version=version)
            )

            action.setCheckable(True)
            if self.version == version and self.installed:
                action.setChecked(True)
                action.setDisabled(True)

            version_actions.append(action)

        install_action = create_action(self.widget, 'Install application', triggered=self.install_application)
        install_action.setEnabled(not self.installed)

        if self.non_conda:
            install_action.setDisabled(True)

        update_action = create_action(self.widget, 'Update application', triggered=self.update_application)

        if versions and versions[-1] == self.version:
            update_action.setDisabled(True)
        else:
            update_action.setDisabled(False)

        remove_action = create_action(self.widget, 'Remove application', triggered=self.remove_application)
        remove_action.setEnabled(self.installed)

        actions = [install_action, update_action, remove_action, None, self.menu_versions]

        if self.non_conda:
            # we're not going to support messing
            # with vscode/pycharm via navigator for now
            update_action.setDisabled(True)
            remove_action.setDisabled(True)
            install_action.setDisabled(True)
            versions = []
            self.menu_versions.setDisabled(True)

        add_actions(self.menu_options, actions)
        add_actions(self.menu_versions, version_actions)
        offset = QPoint(self.button_options.width(), 0)
        position = self.button_options.mapToGlobal(QPoint(0, 0))
        self.menu_versions.setEnabled(len(versions) > 1)
        self.menu_options.move(position + offset)
        self.menu_options.exec_()

    def update_status(self):  # pylint: disable=too-many-branches,too-many-statements
        """Update status."""
        # License check
        self.url = ''
        self.expired = False
        button_label = 'Install'

        # Version and version updates
        if self.versions and self.version != self.versions[-1] and self.installed:
            # The property is used with CSS to display updatable packages.
            self.button_version.setProperty('pressed', True)
            self.button_version.setToolTip(f'Version {self.versions[-1]} available')
        else:
            self.button_version.setProperty('pressed', False)

        # For VScode app do not display if new updates are available
        # See: https://github.com/ContinuumIO/navigator/issues/1504
        if self.non_conda:
            self.button_version.setProperty('pressed', False)
            self.button_version.setToolTip('')

        self.button_install.setText(button_label)
        self.button_install.setVisible(not self.installed)
        self.button_launch.setVisible(self.installed)

        self.button_launch.setEnabled(True)

    def update_versions(self, version=None, versions=None):
        """Update button visibility depending on update availability."""
        if self.installed and version:
            self.button_options.setVisible(True)
            self.button_version.setText(version)
            self.button_version.setVisible(True)
        elif not self.installed and versions:
            self.button_install.setEnabled(True)
            self.button_version.setText(versions[-1])
            self.button_version.setVisible(True)

        self.versions = versions
        self.version = version
        self.update_status()

    def set_loading(self, value):
        """Set loading status."""
        self.button_install.setDisabled(value)
        self.button_options.setDisabled(value)
        self.button_launch.setDisabled(value)

        if value:
            self.spinner.start()
        else:
            self.spinner.stop()
            if self.version is None and self.versions:
                version = self.versions[-1]
            else:
                version = self.version
            self.button_version.setText(version)
            self.button_launch.setDisabled(self.expired)

        self.frame_spinner.setVisible(value)
        self.button_version.setVisible(not value)

    # --- Application actions
    # ------------------------------------------------------------------------
    def install_application(self, value=None, version=None, install=True):  # pylint: disable=unused-argument
        """
        Update the application on the defined prefix environment.

        This is used for both normal install and specific version install.
        """
        telemetry.ANALYTICS.instance.event('install-application', {'application': self.name})
        action = C.APPLICATION_INSTALL if install else C.APPLICATION_UPDATE
        self.widget.sig_conda_action_requested.emit(
            action, self.name, version, C.TAB_HOME, self.non_conda, self.app_type
        )
        if self.app_type == AppType.CONDA:
            self.set_loading(True)

    def remove_application(self):
        """Remove the application from the defined prefix environment."""
        self.widget.sig_conda_action_requested.emit(
            C.APPLICATION_REMOVE, self.name, None, C.TAB_HOME, self.non_conda, self.app_type
        )
        self.set_loading(True)

    def update_application(self):
        """Update the application on the defined prefix environment."""
        # version = None is equivalent to saying
        # "most recent version that is compatible with my env"
        self.install_application(version=None, install=False)

    def launch_application(self) -> None:
        """Launch an application (all types)."""
        telemetry.ANALYTICS.instance.event('launch-application', {'application': self.name})

        command: str
        if self.app_type == AppType.WEB:
            command = external_apps.get_applications(cached=True).web_apps[self.name].url
        else:
            if self.command is None:
                raise ValueError('command not set')

            command = self.command
            update_pointer(Qt.BusyCursor)
            self.button_launch.setDisabled(True)
            self.timer.setInterval(self.timeout)
            self.timer.start()

        self.widget.sig_launch_action_requested.emit(
            self.name,  # package_name
            command,  # command
            self.extra_arguments,
            True,  # leave_path_alone
            self.prefix,  # prefix
            C.TAB_HOME,  # sender
            self.non_conda,  # non_conda
            self.app_type,  # app_type
        )
