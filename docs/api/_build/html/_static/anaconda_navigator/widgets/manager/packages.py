# -*- coding: utf-8 -*-

# pylint: disable=invalid-name,no-name-in-module

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Conda Packager Manager Widget."""

from __future__ import absolute_import, division, print_function, with_statement

import gettext

from qtpy.QtCore import QEvent, QSize, Qt, Signal
from qtpy.QtWidgets import QApplication, QDialogButtonBox, QHBoxLayout, QProgressBar, QPushButton, QVBoxLayout, QWidget

from anaconda_navigator.api.anaconda_api import AnacondaAPI
from anaconda_navigator.config import CONF
from anaconda_navigator.utils import constants as C
from anaconda_navigator.utils import telemetry
from anaconda_navigator.widgets import common as global_commons
from anaconda_navigator.widgets import (
    ButtonBase, ButtonDanger, ButtonNormal, ButtonPrimary, ComboBoxBase, FrameBase, FrameTabFooter, FrameTabHeader,
    LabelBase, SpacerHorizontal,
)
from anaconda_navigator.widgets.helperwidgets import LineEditSearch
from anaconda_navigator.widgets.manager.table import TableCondaPackages


_ = gettext.gettext


# --- Widgtes defined for CSS styling
# -----------------------------------------------------------------------------

class ComboBoxPackageFilter(ComboBoxBase):
    """Combobox used in CSS styling."""

    currentTextChanged = Signal(str)

    def __init__(self, *args, **kwargs):
        """Combobox used in CSS styling."""
        super().__init__(*args, **kwargs)
        self.currentIndexChanged.connect(self._handle_index_changed)

    def _handle_index_changed(self, index):
        """Recreate signal not present in qt4."""
        if isinstance(index, int):
            self.currentTextChanged.emit(self.itemText(index))

    def setCurrentText(self, text):
        """Recreate method not present in qt4."""
        for i in range(self.count()):
            if text == self.itemText(i):
                self.setCurrentIndex(i)
                return


class ButtonPackageChannels(ButtonNormal):
    """Button used in CSS styling."""


class ButtonPackageOk(ButtonNormal):
    """Button used in CSS styling."""


class ButtonPackageApply(ButtonPrimary):
    """Button used in CSS styling."""


class ButtonPackageCancel(ButtonNormal):
    """Button used in CSS styling."""


class ButtonPackageUpdate(ButtonNormal):
    """Button used in CSS styling."""


class ButtonPackageClear(ButtonDanger):
    """Button used in CSS styling."""


class ProgressBarPackage(QProgressBar):  # pylint: disable=too-few-public-methods
    """Progress bar used in CSS styling."""


class LabelPackageStatus(LabelBase):
    """Label used in CSS styling."""


class LabelPackageStatusAction(LabelBase):
    """Label used in CSS styling."""


class FramePackageTop(FrameBase):
    """Frame used in CSS styling."""


class FramePackageBottom(FrameBase):
    """Frame used in CSS styling."""


# --- Navigator helper widgets
# -----------------------------------------------------------------------------

class FirstRowWidget(ButtonBase):
    """Widget located before table to handle focus in and tab focus."""

    sig_enter_first = Signal()

    def __init__(self, widget_before=None):
        """Widget located before table to handle focus in and tab focus."""
        super().__init__()
        self.widget_before = widget_before
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect)  # Needed on mac

    @staticmethod
    def sizeHint():
        """Override Qt method."""
        return QSize(0, 0)

    def focusInEvent(self, event):
        """Override Qt method."""
        self.sig_enter_first.emit()

    def event(self, event):
        """Override Qt method."""
        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key in [Qt.Key_Tab]:
                self.sig_enter_first.emit()
                return True
        return QPushButton.event(self, event)


class LastRowWidget(ButtonBase):
    """Widget located after table to handle focus out and tab focus."""

    sig_enter_last = Signal()

    def __init__(self, widgets_after=None):
        """Widget located after table to handle focus out and tab focus."""
        super().__init__()
        self.widgets_after = widgets_after or []
        self.setAttribute(Qt.WA_LayoutUsesWidgetRect)  # Needed on mac
        self.setFocusPolicy(Qt.StrongFocus)

    def focusInEvent(self, event):
        """Override Qt method."""
        self.sig_enter_last.emit()

    def add_focus_widget(self, widget):
        """Add after focus widget."""
        if widget in self.widgets_after:
            return
        self.widgets_after.append(widget)

    def handle_tab(self):
        """Handle tab focus widget."""
        for w in self.widgets_after:
            if w.isVisible():
                w.setFocus()
                return
        self.setFocus()

    def event(self, event):
        """Override Qt method."""
        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key in [Qt.Key_Backtab]:
                self.sig_enter_last.emit()
                return True
        return QPushButton.event(self, event)

    @staticmethod
    def sizeHint():
        """Override Qt method."""
        return QSize(0, 0)


# --- Main widget
# -----------------------------------------------------------------------------
class CondaPackagesWidget(QWidget):  # pylint: disable=too-many-instance-attributes
    """Conda Packages Widget."""

    sig_ready = Signal()
    sig_next_focus = Signal()

    # conda_packages_action_dict, pip_packages_action_dict
    sig_packages_action_requested = Signal(object, object)

    # button_widget, sender
    sig_channels_requested = Signal(object, object)

    # sender
    sig_update_index_requested = Signal(object)
    sig_cancel_requested = Signal(object)

    def __init__(self, parent, config=CONF):  # pylint: disable=too-many-statements
        """Conda Packages Widget."""
        super().__init__(parent)

        self._parent = parent
        self._current_model_index = None
        self._current_action_name = ''
        self._current_table_scroll = None
        self._hide_widgets = False

        self.api = AnacondaAPI()
        self.prefix = None

        self.message = ''
        self.config = config

        # Widgets
        self.bbox = QDialogButtonBox(Qt.Horizontal)
        self.button_cancel = ButtonPackageCancel('Cancel')
        self.button_channels = ButtonPackageChannels(_('Channels'))
        self.button_ok = ButtonPackageOk(_('Ok'))
        self.button_update = ButtonPackageUpdate(_('Update index...'))
        self.button_apply = ButtonPackageApply(_('Apply'))
        self.button_clear = ButtonPackageClear(_('Clear'))
        self.combobox_filter = ComboBoxPackageFilter(self)
        self.frame_top = FrameTabHeader()
        self.frame_bottom = FrameTabFooter()
        self.progress_bar = ProgressBarPackage(self)
        self.label_status = LabelPackageStatus(self)
        self.label_status_action = LabelPackageStatusAction(self)
        self.table = TableCondaPackages(self)
        self.textbox_search = LineEditSearch(self)
        self.widgets = [
            self.button_update, self.button_channels, self.combobox_filter, self.textbox_search, self.table,
            self.button_ok, self.button_apply, self.button_clear
        ]
        self.table_first_row = FirstRowWidget(widget_before=self.textbox_search)
        self.table_last_row = LastRowWidget(widgets_after=[
            self.button_apply,
            self.button_clear,
            self.button_cancel,
        ])

        # Widgets setup
        max_height = self.label_status.fontMetrics().height()
        max_width = self.textbox_search.fontMetrics().width('0' * 24)
        self.bbox.addButton(self.button_ok, QDialogButtonBox.ActionRole)
        self.button_ok.setMaximumSize(QSize(0, 0))
        self.button_ok.setVisible(False)
        self.button_channels.setCheckable(True)
        self.combobox_filter.addItems(C.COMBOBOX_VALUES_ORDERED[:])
        self.combobox_filter.setMinimumWidth(120)
        self.progress_bar.setMaximumHeight(int(max_height * 1.2))
        self.progress_bar.setMaximumWidth(max_height * 12)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setVisible(False)
        self.setMinimumSize(QSize(480, 300))
        self.setWindowTitle(_('Conda Package Manager'))
        self.label_status.setFixedHeight(int(max_height * 1.5))
        self.textbox_search.setMaximumWidth(max_width)
        self.textbox_search.setMinimumWidth(max_width)
        self.textbox_search.setPlaceholderText('Search Packages')
        self.table_first_row.setMaximumHeight(0)
        self.table_last_row.setMaximumHeight(0)
        self.table_last_row.setVisible(False)
        self.table_first_row.setVisible(False)
        self.te_alert = global_commons.TeamEditionServerAlert()

        # Layout
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.combobox_filter, 0, Qt.AlignCenter)
        top_layout.addWidget(SpacerHorizontal())
        top_layout.addWidget(self.button_channels, 0, Qt.AlignCenter)
        top_layout.addWidget(SpacerHorizontal())
        top_layout.addWidget(self.button_update, 0, Qt.AlignCenter)
        top_layout.addStretch()
        top_layout.addWidget(self.textbox_search, 0, Qt.AlignCenter)
        self.frame_top.setLayout(top_layout)

        middle_layout = QVBoxLayout()
        middle_layout.addWidget(self.table_first_row)
        middle_layout.addWidget(self.te_alert)
        middle_layout.addWidget(self.table)
        middle_layout.addWidget(self.table_last_row)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.label_status_action)
        bottom_layout.addWidget(SpacerHorizontal())
        bottom_layout.addWidget(self.label_status)
        bottom_layout.addStretch()
        bottom_layout.addWidget(self.progress_bar)
        bottom_layout.addWidget(SpacerHorizontal())
        bottom_layout.addWidget(self.button_cancel)
        bottom_layout.addWidget(SpacerHorizontal())
        bottom_layout.addWidget(self.button_apply)
        bottom_layout.addWidget(SpacerHorizontal())
        bottom_layout.addWidget(self.button_clear)
        self.frame_bottom.setLayout(bottom_layout)

        layout = QVBoxLayout(self)
        layout.addWidget(self.frame_top)
        layout.addLayout(middle_layout)
        layout.addWidget(self.frame_bottom)
        self.setLayout(layout)

        # Signals and slots
        self.button_cancel.clicked.connect(lambda: self.sig_cancel_requested.emit(C.TAB_ENVIRONMENT))
        self.combobox_filter.currentTextChanged.connect(self.filter_package)
        self.button_apply.clicked.connect(self.apply_multiple_actions)
        self.button_clear.clicked.connect(self.clear_actions)
        self.button_channels.clicked.connect(self.show_channels)
        self.button_update.clicked.connect(self.update_package_index)
        self.textbox_search.textChanged.connect(self.search_package)
        self.table.sig_actions_updated.connect(self.update_actions)
        self.table.sig_status_updated.connect(self.update_status)

        self.table.sig_next_focus.connect(self.table_last_row.handle_tab)
        self.table.sig_previous_focus.connect(self.table_first_row.widget_before.setFocus)
        self.table_first_row.sig_enter_first.connect(self._handle_tab_focus)
        self.table_last_row.sig_enter_last.connect(self._handle_backtab_focus)

        self.button_cancel.setVisible(False)

    # --- Helpers
    # -------------------------------------------------------------------------
    def _handle_tab_focus(self):
        self.table.setFocus()
        if self.table.proxy_model:
            index = self.table.proxy_model.index(0, 0)
            self.table.setCurrentIndex(index)

    def _handle_backtab_focus(self):
        self.table.setFocus()
        if self.table.proxy_model:
            row = self.table.proxy_model.rowCount() - 1
            index = self.table.proxy_model.index(row, 0)
            self.table.setCurrentIndex(index)

    # --- Setup
    # -------------------------------------------------------------------------
    def setup(self, packages=None, model_data=None, prefix=None):  # pylint: disable=unused-argument
        """
        Setup packages.

        Populate the table with `packages` information.

        Parameters
        ----------
        packages: dict
            Grouped package information by package name.
        blacklist: list of str
            List of conda package names to be excluded from the actual package
            manager view.
        """
        self.table.setup_model(packages, model_data)
        combobox_text = self.combobox_filter.currentText()
        self.combobox_filter.setCurrentText(combobox_text)
        self.filter_package(combobox_text)
        self.table.setFocus()
        self.sig_ready.emit()

    # --- Other methods
    # -------------------------------------------------------------------------
    def apply_multiple_actions(self):
        """Apply multiple actions on packages."""
        actions = self.table.get_actions()
        pip_actions = actions[C.PIP_PACKAGE]
        conda_actions = actions[C.CONDA_PACKAGE]

        self.sig_packages_action_requested.emit(conda_actions, pip_actions)

    def clear_actions(self):
        """Clear the table actions."""
        self.table.clear_actions()

    def filter_package(self, value):
        """Filter packages by type."""
        self.table.filter_status_changed(value)

    def search_package(self, text):
        """Search and filter packages by text."""
        self.table.search_string_changed(text)

    def show_channels(self):
        """Show channel dialog."""
        self.sig_channels_requested.emit(
            self.button_channels,
            C.TAB_ENVIRONMENT,
        )

    def update_actions(self, number_of_actions):
        """Update visibility of buttons based on actions."""
        self.button_apply.setVisible(bool(number_of_actions))
        self.button_clear.setVisible(bool(number_of_actions))

    def update_package_index(self):
        """Update pacakge index."""
        telemetry.ANALYTICS.instance.event('update-index')
        self.sig_update_index_requested.emit(C.ENVIRONMENT_PACKAGE_MANAGER)

    # --- Common methods
    # -------------------------------------------------------------------------
    def ordered_widgets(self):  # pylint: disable=missing-function-docstring
        pass

    def set_widgets_enabled(self, value):
        """Set the enabled status of widgets and subwidgets."""
        self.table.setEnabled(value)
        self.button_clear.setEnabled(value)
        self.button_apply.setEnabled(value)
        self.button_cancel.setEnabled(not value)
        self.button_cancel.setVisible(not value)

    def update_status(self, action='', message='', value=None, max_value=None):
        """
        Update status of package widget.

        - progress == None and max_value == None -> Not Visible
        - progress == 0 and max_value == 0 -> Busy
        - progress == n and max_value == m -> Progress values
        """
        # Elide if too big
        width = QApplication.desktop().availableGeometry().width()
        max_status_length = round(width * (1.0 / 2.0), 0)
        msg_percent = 0.70

        fm = self.label_status_action.fontMetrics()
        action = fm.elidedText(action, Qt.ElideRight, round(max_status_length * msg_percent))
        message = fm.elidedText(message, Qt.ElideRight, round(max_status_length * (1 - msg_percent)))

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
