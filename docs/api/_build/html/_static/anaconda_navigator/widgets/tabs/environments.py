# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Environments Tab."""

from __future__ import absolute_import, division, print_function

from qtpy.QtCore import QPoint, Qt, Signal  # pylint: disable=no-name-in-module
from qtpy.QtWidgets import QHBoxLayout, QMenu, QVBoxLayout  # pylint: disable=no-name-in-module
from anaconda_navigator.api.anaconda_api import AnacondaAPI
from anaconda_navigator.utils import constants as C
from anaconda_navigator.utils import launch
from anaconda_navigator.widgets import (
    ButtonToolNormal, FrameEnvironmentsList, FrameEnvironmentsPackages, FrameTabHeader, WidgetBase,
)
from anaconda_navigator.widgets.helperwidgets import ButtonToggleCollapse, LineEditSearch
from anaconda_navigator.widgets.lists.environments import ListItemEnv, ListWidgetEnv
from anaconda_navigator.widgets.manager.packages import CondaPackagesWidget


class EnvironmentsTab(WidgetBase):  # pylint: disable=too-many-instance-attributes
    """Conda environments tab."""
    BLACKLIST = ['anaconda-navigator']  # Hide in package manager; pylint: disable=invalid-name

    # --- Signals
    # -------------------------------------------------------------------------
    sig_ready = Signal()

    # name, prefix, sender
    sig_item_selected = Signal(object, object, object)

    # sender, func_after_dlg_accept, func_callback_on_finished
    sig_create_requested = Signal()
    sig_clone_requested = Signal()
    sig_backup_requested = Signal()
    sig_import_requested = Signal()
    sig_remove_requested = Signal()

    # button_widget, sender_constant
    sig_channels_requested = Signal(object, object)

    # sender_constant
    sig_update_index_requested = Signal(object)
    sig_cancel_requested = Signal(object)

    # conda_packages_action_dict, pip_packages_action_dict
    sig_packages_action_requested = Signal(object, object)

    def __init__(self, parent=None):  # pylint: disable=too-many-statements
        """Conda environments tab."""
        super().__init__(parent)

        # Variables
        self.api = AnacondaAPI()
        self.current_prefix = None

        # Widgets
        self.frame_header_left = FrameTabHeader()
        self.frame_list = FrameEnvironmentsList(self)
        self.frame_widget = FrameEnvironmentsPackages(self)
        self.text_search = LineEditSearch()
        self.list = ListWidgetEnv()
        self.menu_list = QMenu()
        self.button_create = ButtonToolNormal(text='Create')
        self.button_clone = ButtonToolNormal(text='Clone')
        self.button_import = ButtonToolNormal(text='Import')
        self.button_backup = ButtonToolNormal(text='Backup')
        self.button_remove = ButtonToolNormal(text='Remove')
        self.button_toggle_collapse = ButtonToggleCollapse()
        self.widget = CondaPackagesWidget(parent=self)

        # Widgets setup
        self.frame_list.is_expanded = True
        self.text_search.setPlaceholderText('Search Environments')
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.button_create.setObjectName('create')  # Needed for QSS selectors
        self.button_clone.setObjectName('clone')
        self.button_import.setObjectName('import')
        self.button_backup.setObjectName('backup')
        self.button_remove.setObjectName('remove')
        self.widget.textbox_search.set_icon_visibility(False)

        # Layouts
        layout_header_left = QVBoxLayout()
        layout_header_left.addWidget(self.text_search)
        self.frame_header_left.setLayout(layout_header_left)

        layout_buttons = QHBoxLayout()
        layout_buttons.addWidget(self.button_create)
        layout_buttons.addWidget(self.button_clone)
        layout_buttons.addWidget(self.button_import)
        layout_buttons.addWidget(self.button_backup)
        layout_buttons.addWidget(self.button_remove)

        layout_list_buttons = QVBoxLayout()
        layout_list_buttons.addWidget(self.frame_header_left)
        layout_list_buttons.addWidget(self.list)
        layout_list_buttons.addLayout(layout_buttons)
        self.frame_list.setLayout(layout_list_buttons)

        layout_widget = QHBoxLayout()
        layout_widget.addWidget(self.widget)
        self.frame_widget.setLayout(layout_widget)

        layout_main = QHBoxLayout()
        layout_main.addWidget(self.frame_list, 10)
        layout_main.addWidget(self.button_toggle_collapse, 1)
        layout_main.addWidget(self.frame_widget, 30)

        self.setLayout(layout_main)

        # Signals for buttons and boxes
        self.button_toggle_collapse.clicked.connect(self.expand_collapse)
        self.button_create.clicked.connect(self.sig_create_requested)
        self.button_clone.clicked.connect(self.sig_clone_requested)
        self.button_import.clicked.connect(self.sig_import_requested)
        self.button_backup.clicked.connect(self.sig_backup_requested)
        self.button_remove.clicked.connect(self.sig_remove_requested)
        self.text_search.textChanged.connect(self.filter_list)

        # Signals for list
        self.list.sig_item_selected.connect(self._item_selected)

        # Signals for packages widget
        self.widget.sig_ready.connect(self.sig_ready)
        self.widget.sig_channels_requested.connect(self.sig_channels_requested)
        self.widget.sig_update_index_requested.connect(self.sig_update_index_requested)
        self.widget.sig_cancel_requested.connect(self.sig_cancel_requested)
        self.widget.sig_packages_action_requested.connect(self.sig_packages_action_requested)

    # --- Setup methods
    # -------------------------------------------------------------------------
    def setup(self, conda_data):
        """Setup tab content and populates the list of environments."""
        self.set_widgets_enabled(False)
        conda_processed_info = conda_data.get('processed_info')
        environments = conda_processed_info.get('__environments')
        packages = conda_data.get('packages')
        self.current_prefix = conda_processed_info.get('default_prefix')
        self.set_environments(environments)
        self.set_packages(packages)

    def set_environments(self, environments):
        """Populate the list of environments."""
        self.list.clear()
        selected_item_row = 0
        for i, (env_prefix, env_name) in enumerate(environments.items()):  # pylint: disable=invalid-name
            item = ListItemEnv(prefix=env_prefix, name=env_name)
            item.button_options.clicked.connect(self.show_environment_menu)
            if env_prefix == self.current_prefix:
                selected_item_row = i
            self.list.addItem(item)

        self.list.setCurrentRow(selected_item_row, loading=True)
        self.filter_list()
        self.list.scrollToItem(self.list.item(selected_item_row))

    def _set_packages(self, worker, output, error):  # pylint: disable=unused-argument
        """Set packages callback."""
        packages, model_data = output
        self.widget.setup(packages, model_data)
        self.set_widgets_enabled(True)
        self.set_loading(prefix=self.current_prefix, value=False)

    def set_packages(self, packages):
        """Set packages widget content."""
        worker = self.api.process_packages(packages, prefix=self.current_prefix, blacklist=self.BLACKLIST)
        worker.sig_chain_finished.connect(self._set_packages)

    def show_environment_menu(self, value=None, position=None):  # pylint: disable=unused-argument
        """Show the environment actions menu."""
        self.menu_list.clear()
        menu_item = self.menu_list.addAction('Open Terminal')
        menu_item.triggered.connect(lambda: self.open_environment_in('terminal'))

        for word in ['Python', 'IPython', 'Jupyter Notebook']:
            menu_item = self.menu_list.addAction('Open with ' + word)
            menu_item.triggered.connect(lambda x, w=word: self.open_environment_in(w.lower()))

        current_item = self.list.currentItem()
        prefix = current_item.prefix

        if isinstance(position, bool) or position is None:
            width = current_item.button_options.width()
            position = QPoint(width, 0)

        point = QPoint(0, 0)
        parent_position = current_item.button_options.mapToGlobal(point)
        self.menu_list.move(parent_position + position)

        # Disabled actions depending on the environment installed packages
        actions = self.menu_list.actions()
        actions[2].setEnabled(launch.check_prog('ipython', prefix))
        actions[3].setEnabled(launch.check_prog('notebook', prefix))

        self.menu_list.exec_()

    def open_environment_in(self, which):
        """Open selected environment in console terminal."""
        prefix = self.list.currentItem().prefix

        if which == 'terminal':
            launch.console(prefix)
        else:
            launch.py_in_console(prefix, which)

    # --- Common Helpers (# NOTE: factor out to common base widget)
    # -------------------------------------------------------------------------
    def _item_selected(self, item):
        """Callback to emit signal as user selects an item from the list."""
        self.set_loading(prefix=item.prefix)
        self.sig_item_selected.emit(item.name, item.prefix, C.TAB_ENVIRONMENT)

    def add_temporal_item(self, name):
        """Creates a temporal item on list while creation becomes effective."""
        item_names = [item.name for item in self.list.items()]
        item_names.append(name)
        index = list(sorted(item_names)).index(name) + 1
        item = ListItemEnv(name=name)
        self.list.insertItem(index, item)
        self.list.setCurrentRow(index)
        self.list.scrollToItem(item)
        item.set_loading(True)

    def expand_collapse(self):
        """Expand or collapse the list selector."""
        if self.frame_list.is_expanded:
            self.frame_list.hide()
            self.frame_list.is_expanded = False
        else:
            self.frame_list.show()
            self.frame_list.is_expanded = True

    def filter_list(self, text=None):
        """Filter items in list by name."""
        text = self.text_search.text().lower()
        for i in range(self.list.count()):  # pylint: disable=invalid-name
            item = self.list.item(i)
            item.setHidden(text not in item.name.lower())

            if not item.widget.isVisible():
                item.widget.repaint()

    def ordered_widgets(self, next_widget=None):
        """Return a list of the ordered widgets."""
        if next_widget is not None:
            self.widget.table_last_row.add_focus_widget(next_widget)

        ordered_widgets = [self.text_search]
        ordered_widgets += self.list.ordered_widgets()
        ordered_widgets += [
            self.button_create,
            self.button_clone,
            self.button_import,
            self.button_backup,
            self.button_remove,
            self.widget.combobox_filter,
            self.widget.button_channels,
            self.widget.button_update,
            self.widget.textbox_search,
            # self.widget.table_first_row,
            self.widget.table,
            self.widget.table_last_row,
            self.widget.button_apply,
            self.widget.button_clear,
            self.widget.button_cancel,
        ]
        return ordered_widgets

    def refresh(self):
        """Refresh the enabled/disabled status of the widget and subwidgets."""
        is_root = self.current_prefix == self.api.ROOT_PREFIX
        self.button_clone.setDisabled(is_root)
        self.button_remove.setDisabled(is_root)

    def set_loading(self, prefix=None, value=True):
        """Set the item given by `prefix` to loading state."""
        for row, item in enumerate(self.list.items()):
            if item.prefix == prefix:
                item.set_loading(value)
                self.list.setCurrentRow(row)
                break

    def set_widgets_enabled(self, value):
        """Change the enabled status of widgets and subwidgets."""
        self.list.setEnabled(value)
        self.button_create.setEnabled(value)
        self.button_clone.setEnabled(value)
        self.button_import.setEnabled(value)
        self.button_backup.setEnabled(value)
        self.button_remove.setEnabled(value)
        self.widget.set_widgets_enabled(value)
        if value:
            self.refresh()

    def update_status(self, action='', message='', value=None, max_value=None):
        """Update widget status and progress bar."""
        self.widget.update_status(action=action, message=message, value=value, max_value=max_value)

    def update_style_sheet(self):
        """Update custom CSS stylesheet."""
        self.list.update_style_sheet()
