# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""
Home Tab.

This widget does not perform the actual actions but it emits signals that
should be connected to the final controller on the main window.
"""

from __future__ import absolute_import, division, print_function

__all__ = ['HomeTab']

import typing

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QApplication, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout

from anaconda_navigator.api.anaconda_api import AnacondaAPI
from anaconda_navigator.api import types as api_types
from anaconda_navigator.utils import constants
from anaconda_navigator.utils import telemetry
from anaconda_navigator.widgets import common as global_commons
from anaconda_navigator.widgets import (
    ButtonNormal, ComboBoxBase, FrameTabContent, FrameTabFooter, FrameTabHeader, LabelBase, SpacerHorizontal, WidgetBase
)
from anaconda_navigator.widgets.lists.apps import ListItemApplication, ListWidgetApplication


ApplicationFilter = typing.Literal['all', 'installed', 'uninstalled', 'updatable']


# --- Application filters
# -----------------------------------------------------------------------------

class ApplicationFilterDetails(typing.NamedTuple):
    """Additional details for the application filters."""

    verbose_name: str

    filter_function: typing.Callable[['api_types.Application'], bool]


class CheckInstalled:  # pylint: disable=too-few-public-methods
    """
    Common checker for application installation state.

    :param target: Expected installation state of an application.
    """

    __slots__ = ('__target',)

    def __init__(self, target: bool) -> None:
        """Initialize new :class:`~CheckInstalled` instance."""
        self.__target: typing.Final[bool] = target

    @property
    def target(self) -> bool:  # noqa: D401
        """Expected installation state."""
        return self.__target

    def __call__(self, application: 'api_types.Application') -> bool:
        """Check if application installation state."""
        return application.get('installed', False) is self.__target


def check_updatable(application: 'api_types.Application') -> bool:
    """Check if there is an update for an `application`."""
    version: typing.Optional[str] = application.get('version', None)
    versions: typing.Sequence[str] = application.get('versions', ())
    return application.get('installed', False) and bool(version) and bool(versions) and (version != versions[-1])


APPLICATION_FILTERS: typing.Final[typing.Mapping[ApplicationFilter, ApplicationFilterDetails]] = {
    'all': ApplicationFilterDetails('All applications', lambda application: True),
    'installed': ApplicationFilterDetails('Installed applications', CheckInstalled(True)),
    'uninstalled': ApplicationFilterDetails('Not installed applications', CheckInstalled(False)),
    'updatable': ApplicationFilterDetails('Updatable applications', check_updatable),
}


def application_sorting_key(application: 'api_types.Application') -> typing.Tuple[int, bool, int, str]:
    """Prepare sorting key for an application."""
    rank: int = -application.get('rank', 0)
    installed: bool = not application.get('installed', False)

    category: int
    application_type: typing.Optional[constants.AppType] = application.get('app_type', None)
    if application_type in (constants.AppType.INSTALLABLE, constants.AppType.CONDA):
        category = 0
    elif application_type == constants.AppType.WEB:
        category = 1
    else:
        category = 2

    display_name: str = application.get('display_name', '').lower()

    return rank, installed, category, display_name


# --- Custom widgets used with CSS styling
# -----------------------------------------------------------------------------

class ButtonHomeRefresh(global_commons.IconButton):  # pylint: disable=too-few-public-methods
    """QFrame used for CSS styling refresh button inside the Home Tab."""


class ComboHomeFilter(ComboBoxBase):  # pylint: disable=too-few-public-methods
    """Widget Used for CSS styling."""


class ComboHomeEnvironment(ComboBoxBase):  # pylint: disable=too-few-public-methods
    """Widget Used for CSS styling."""


class ButtonHomeChannels(ButtonNormal):
    """Widget Used for CSS styling."""


class LabelHome(LabelBase):
    """QLabel used for CSS styling the Home Tab label."""


# --- Main widget
# -----------------------------------------------------------------------------
class HomeTab(WidgetBase):  # pylint: disable=too-many-instance-attributes
    """Home applications tab."""
    # name, prefix, sender
    sig_item_selected = Signal(object, object, object)

    # button_widget, sender
    sig_channels_requested = Signal(object, object)

    # application_name, command, extra_arguments, prefix, leave_path_alone, sender, non_conda, app_type
    sig_launch_action_requested = Signal(object, object, object, bool, object, object, object, object)

    # action, application_name, version, sender, non_conda, app_type
    sig_conda_action_requested = Signal(object, object, object, object, object, object)

    # url
    sig_url_clicked = Signal(object)

    # NOTE: Connect these signals to have more granularity
    # [{'name': package_name, 'version': version}...], sender
    sig_install_action_requested = Signal(object, object)
    sig_remove_action_requested = Signal(object, object)

    def __init__(self, parent=None):  # pylint: disable=too-many-statements
        """Home applications tab."""
        super().__init__(parent)

        # Variables
        self._parent = parent
        self.api = AnacondaAPI()
        self.applications = None
        self.app_timers = None
        self.current_prefix = None

        self.__applications: typing.Mapping['api_types.ApplicationName', 'api_types.Application'] = {}
        self.__applications_filter: 'ApplicationFilter' = 'all'

        # Widgets
        self.list = ListWidgetApplication()
        self.button_channels = ButtonHomeChannels('Channels')
        self.button_refresh = ButtonHomeRefresh()
        self.combo_filter = ComboHomeFilter()
        self.label_home = LabelHome('on')
        self.combo_environment = ComboHomeEnvironment()
        self.frame_top = FrameTabHeader(self)
        self.frame_body = FrameTabContent(self)
        self.frame_bottom = FrameTabFooter(self)
        self.label_status_action = QLabel('')
        self.label_status = QLabel('')
        self.progress_bar = QProgressBar()
        self.first_widget = self.combo_environment
        self.te_alert = global_commons.TeamEditionServerAlert()

        # Widget setup
        self.setObjectName('Tab')
        self.progress_bar.setTextVisible(False)
        self.list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        application_filter: 'ApplicationFilter'
        application_filter_details: ApplicationFilterDetails
        for application_filter, application_filter_details in APPLICATION_FILTERS.items():
            self.combo_filter.addItem(application_filter_details.verbose_name, application_filter)
        self.combo_filter.setCurrentText(APPLICATION_FILTERS[self.__applications_filter].verbose_name)

        # Layout
        layout_top = QHBoxLayout()
        layout_top.addWidget(self.combo_filter)
        layout_top.addWidget(SpacerHorizontal())
        layout_top.addWidget(self.label_home)
        layout_top.addWidget(SpacerHorizontal())
        layout_top.addWidget(self.combo_environment)
        layout_top.addWidget(SpacerHorizontal())
        layout_top.addWidget(self.button_channels)
        layout_top.addWidget(SpacerHorizontal())
        layout_top.addStretch()
        layout_top.addWidget(self.button_refresh)
        self.frame_top.setLayout(layout_top)

        layout_body = QVBoxLayout()
        layout_body.addWidget(self.te_alert)
        layout_body.addWidget(self.list)
        self.frame_body.setLayout(layout_body)

        layout_bottom = QHBoxLayout()
        layout_bottom.addWidget(self.label_status_action)
        layout_bottom.addWidget(SpacerHorizontal())
        layout_bottom.addWidget(self.label_status)
        layout_bottom.addStretch()
        layout_bottom.addWidget(self.progress_bar)
        self.frame_bottom.setLayout(layout_bottom)

        layout = QVBoxLayout()
        layout.addWidget(self.frame_top)
        layout.addWidget(self.frame_body)
        layout.addWidget(self.frame_bottom)
        self.setLayout(layout)

        # Signals
        self.combo_filter.currentIndexChanged.connect(self._filter_selected)
        self.list.sig_conda_action_requested.connect(self.sig_conda_action_requested)
        self.list.sig_url_clicked.connect(self.sig_url_clicked)
        self.list.sig_launch_action_requested.connect(self.sig_launch_action_requested)
        self.button_channels.clicked.connect(self.show_channels)
        self.button_refresh.clicked.connect(self.refresh_cards)
        self.progress_bar.setVisible(False)

    # --- Setup methods
    # -------------------------------------------------------------------------
    def setup(self, conda_data):
        """Setup the tab content."""
        conda_processed_info = conda_data.get('processed_info')
        environments = conda_processed_info.get('__environments')
        self.current_prefix = conda_processed_info.get('default_prefix')
        self.set_environments(environments)

        self.__applications = conda_data.get('applications')
        self.update_applications()

    def set_environments(self, environments):
        """Setup the environments list."""
        # Disconnect to avoid triggering the signal when updating the content
        try:
            self.combo_environment.currentIndexChanged.disconnect()
        except TypeError:
            pass

        self.combo_environment.clear()

        font_metrics = self.combo_environment.fontMetrics()
        widths = []

        index: int
        for index, (env_prefix, env_name) in enumerate(environments.items()):
            widths.append(font_metrics.width(env_name))
            self.combo_environment.addItem(env_name, env_prefix)
            self.combo_environment.setItemData(index, env_prefix, Qt.ToolTipRole)

        for index, (env_prefix, env_name) in enumerate(environments.items()):
            if self.current_prefix == env_prefix:
                break
        else:
            index = 0

        self.combo_environment.setCurrentIndex(index)
        self.combo_environment.currentIndexChanged.connect(self._environment_selected)

        # Fix combobox width
        width = max(widths) + 64
        self.combo_environment.setMinimumWidth(width)

    def update_applications(self) -> None:
        """Build the list of applications present in the current conda env."""
        processed_applications: typing.List['api_types.Application'] = sorted(
            filter(
                APPLICATION_FILTERS[self.__applications_filter].filter_function,
                self.api.process_apps(self.__applications, prefix=self.current_prefix).values(),
            ),
            key=application_sorting_key,
        )

        self.list.clear()

        application_data: 'api_types.Application'
        for application_data in processed_applications:
            self.list.addItem(ListItemApplication(prefix=self.current_prefix, **application_data))

        self.list.update_style_sheet()
        self.set_widgets_enabled(True)
        self.update_status()

    # --- Other methods
    # -------------------------------------------------------------------------
    def current_environment(self):
        """Return the current selected environment."""
        env_name: str = self.combo_environment.currentText()
        return self.api.conda_get_prefix_envname(env_name)

    def refresh_cards(self):
        """Refresh application widgets.

        List widget items sometimes are hidden on resize. This method tries
        to compensate for that refreshing and repainting on user demand.
        """
        telemetry.ANALYTICS.instance.event('refresh-applications')
        self.list.update_style_sheet()
        self.list.repaint()
        for item in self.list.items():
            if not item.widget.isVisible():
                item.widget.repaint()

        worker = self.api.conda_data(prefix=self.current_prefix)
        worker.sig_chain_finished.connect(lambda _, output, __: self.setup(output))
        self.update_status(action='Refreshing applications', value=0, max_value=0)

    def show_channels(self):
        """Emit signal requesting the channels dialog editor."""
        self.sig_channels_requested.emit(self.button_channels, constants.TAB_HOME)

    # --- Common Helpers (# NOTE: factor out to common base widget)
    # -------------------------------------------------------------------------
    def _environment_selected(self, index: int) -> None:
        """Notify that the item in combo (environment) changed."""
        name: str = self.combo_environment.itemText(index)
        prefix: str = self.combo_environment.itemData(index)
        self.sig_item_selected.emit(name, prefix, constants.TAB_HOME)

    def _filter_selected(self, index: int) -> None:
        """Process change of application filter."""
        self.applications_filter = self.combo_filter.itemData(index)

    @property
    def last_widget(self):
        """Return the last element of the list to be used in tab ordering."""
        if self.list.items():
            return self.list.items()[-1].widget
        return None

    @property
    def applications_filter(self) -> 'ApplicationFilter':  # noqa: D401
        """Current filter applied to the application list."""
        return self.__applications_filter

    @applications_filter.setter
    def applications_filter(self, value: 'ApplicationFilter') -> None:
        """Update `application_filter` value."""
        if self.__applications_filter == value:
            return

        self.combo_filter.setCurrentText(APPLICATION_FILTERS[value].verbose_name)
        self.__applications_filter = value
        self.update_applications()

    def ordered_widgets(self, next_widget=None):  # pylint: disable=unused-argument
        """Return a list of the ordered widgets."""
        ordered_widgets = [
            self.combo_filter,
            self.combo_environment,
            self.button_channels,
            self.button_refresh,
        ]
        ordered_widgets += self.list.ordered_widgets()

        return ordered_widgets

    def set_widgets_enabled(self, value: bool) -> None:
        """Enable or disable widgets."""
        self.combo_filter.setEnabled(value)
        self.combo_environment.setEnabled(value)
        self.button_channels.setEnabled(value)
        self.button_refresh.setEnabled(value)

        for item in self.list.items():
            item.button_install.setEnabled(value)
            item.button_options.setEnabled(value)

            if value:
                item.set_loading(not value)

    def update_items(self):
        """Update status of items in list."""
        if self.list:
            for item in self.list.items():
                item.update_status()

    def update_status(self, action='', message='', value=None, max_value=None):
        """Update the application action status."""

        # Elide if too big
        width = QApplication.desktop().availableGeometry().width()
        max_status_length = round(width * (2.0 / 3.0), 0)
        msg_percent = 0.70

        font_metrics = self.label_status_action.fontMetrics()
        action = font_metrics.elidedText(action, Qt.ElideRight, round(max_status_length * msg_percent))
        message = font_metrics.elidedText(message, Qt.ElideRight, round(max_status_length * (1 - msg_percent)))
        self.label_status_action.setText(action)
        self.label_status.setText(message)

        if max_value is None and value is None:
            self.progress_bar.setVisible(False)
        else:
            self.progress_bar.setVisible(True)
            self.progress_bar.setMaximum(max_value)
            self.progress_bar.setValue(value)

    def update_style_sheet(self):
        """Update custom CSS style sheet."""
        self.list.update_style_sheet()
