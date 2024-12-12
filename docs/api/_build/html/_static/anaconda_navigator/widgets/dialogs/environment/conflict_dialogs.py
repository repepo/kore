# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Dialogs for conflicting situations."""

__all__ = ['ConflictDialog']

from qtpy import QtCore
from qtpy import QtWidgets
from anaconda_navigator import widgets
from . import common


class ConflictDialog(common.EnvironmentActionsDialog):  # pylint: disable=too-many-instance-attributes
    """Create new environment dialog if navigator conflicts with deps."""

    def __init__(self, parent=None, package=None, extra_message='', current_prefix=None):
        """Create new environment dialog if navigator conflicts with deps."""
        super().__init__(parent=parent)

        parts = package.split('=')
        self.package = parts[0] if '=' in package else package
        self.package_version = parts[-1] if '=' in package else ''
        self.current_prefix = current_prefix

        base_message = f'<b>{package}</b> cannot be installed on this environment.'

        base_message = extra_message or base_message
        # Widgets
        self.label_info = widgets.LabelBase(
            base_message + '<br><br>'
            'Do you want to install the package in an existing '
            'environment or <br>create a new environment?'
        )
        self.label_name = widgets.LabelBase('Name:')
        self.label_prefix = widgets.LabelBase(' ' * 100)
        self.label_location = widgets.LabelBase('Location:')
        self.combo_name = widgets.ComboBoxBase()
        self.button_ok = widgets.ButtonPrimary('Create')
        self.button_cancel = widgets.ButtonNormal('Cancel')

        # Widgets setup
        self.align_labels([self.label_name, self.label_location])
        self.combo_name.setEditable(True)
        self.combo_name.setCompleter(None)
        self.setMinimumWidth(self.BASE_DIALOG_WIDTH)
        self.setWindowTitle(f'Create new environment for \'{package}\'')
        self.label_prefix.setObjectName('environment-location')
        self.combo_name.setObjectName('environment-selection')

        # Layouts
        grid_layout = QtWidgets.QGridLayout()
        grid_layout.addWidget(self.label_name, 0, 0)
        grid_layout.addWidget(widgets.SpacerHorizontal(), 0, 1)
        grid_layout.addWidget(self.combo_name, 0, 2)
        grid_layout.addWidget(widgets.SpacerVertical(), 1, 0)
        grid_layout.addWidget(self.label_location, 2, 0)
        grid_layout.addWidget(widgets.SpacerHorizontal(), 2, 1)
        grid_layout.addWidget(self.label_prefix, 2, 2)

        layout_buttons = QtWidgets.QHBoxLayout()
        layout_buttons.addStretch()
        layout_buttons.addWidget(self.button_cancel)
        layout_buttons.addWidget(widgets.SpacerHorizontal())
        layout_buttons.addWidget(self.button_ok)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.label_info)
        main_layout.addWidget(widgets.SpacerVertical())
        main_layout.addWidget(widgets.SpacerVertical())
        main_layout.addLayout(grid_layout)
        main_layout.addWidget(widgets.SpacerVertical())
        main_layout.addWidget(widgets.SpacerVertical())
        main_layout.addLayout(layout_buttons)

        self.setLayout(main_layout)

        # Signals
        self.button_ok.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)
        self.combo_name.setCurrentText(self.package)
        self.combo_name.currentTextChanged.connect(self.refresh)
        self.button_ok.setDisabled(True)

    def new_env_name(self):
        """Generate a unique environment name."""
        pkg_name_version = self.package + '-' + self.package_version
        if self.environments:
            if self.package not in self.environments.values():
                env_name = self.package
            elif pkg_name_version not in self.environments.values():
                env_name = pkg_name_version
            else:
                for i in range(1, 1000):  # pylint: disable=invalid-name
                    new_pkg_name = pkg_name_version + '_' + str(i)
                    if new_pkg_name not in self.environments.values():
                        env_name = new_pkg_name
                        break
        else:
            env_name = self.package
        return env_name

    def setup(self, worker, info, error):
        """Setup the dialog conda information as a result of a worker."""
        super().setup(worker, info, error)
        self.combo_name.blockSignals(True)
        self.combo_name.clear()
        new_env_name = self.new_env_name()
        self.combo_name.addItem(new_env_name, new_env_name)
        for i, (env_prefix, env_name) in enumerate(self.environments.items()):  # pylint: disable=invalid-name
            # Do not include the env where the conflict was found!
            if self.current_prefix != env_prefix:
                self.combo_name.addItem(env_name, env_prefix)
                self.combo_name.setItemData(i, env_prefix, QtCore.Qt.ToolTipRole)
        self.combo_name.setCurrentText(new_env_name)
        self.combo_name.blockSignals(False)
        self.refresh()

    def refresh(self):
        """Refresh state of buttons based on content."""
        self.update_location()

        if self.environments:
            self.button_ok.setEnabled(bool(self.name))

    @property
    def name(self):
        """Return the name of the environment."""
        return self.combo_name.currentText()
