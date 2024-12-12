# -*- coding: utf-8 -*-

# pylint: disable=invalid-name,no-name-in-module

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Conda packages table view."""

from __future__ import division, print_function, unicode_literals, with_statement

import gettext
from qtpy import PYQT4, PYQT5
from qtpy.QtCore import QEvent, QPoint, QSize, Qt, QUrl, Signal
from qtpy.QtGui import QColor, QDesktopServices, QIcon, QPen
from qtpy.QtWidgets import QAbstractItemView, QHeaderView, QItemDelegate, QMenu, QTableView
from anaconda_navigator.config import MAC
from anaconda_navigator.utils import constants as C
from anaconda_navigator.utils import get_image_path
from anaconda_navigator.utils import telemetry
from anaconda_navigator.utils.py3compat import to_text_string
from anaconda_navigator.utils.qthelpers import add_actions, create_action
from anaconda_navigator.widgets.manager.filter import MultiColumnSortFilterProxy
from anaconda_navigator.widgets.manager.model import CondaPackagesModel


# --- Local
# -----------------------------------------------------------------------------

_ = gettext.gettext

# --- Constants
# -----------------------------------------------------------------------------
HIDE_COLUMNS = [
    C.COL_STATUS,
    C.COL_URL,
    C.COL_ACTION_VERSION,
]


class CustomDelegate(QItemDelegate):
    """Custom delegate to handle selected/hovered behavior of rows."""
    def paint(self, painter, option, index):
        """Override Qt method."""
        QItemDelegate.paint(self, painter, option, index)
        column = index.column()
        row = index.row()
        rect = option.rect

        if column in [C.COL_NAME, C.COL_DESCRIPTION, C.COL_VERSION]:
            pen = QPen()
            pen.setWidth(1)
            pen.setColor(QColor('#ddd'))
            painter.setPen(pen)
            painter.drawLine(rect.topRight(), rect.bottomRight())

        if (row == self.current_hover_row() or row == self.current_row() and self.has_focus_or_context()):
            pen = QPen()
            pen.setWidth(1)
            if row == self.current_row():
                pen.setColor(QColor('#007041'))
            else:
                pen.setColor(QColor('#43b02a'))
            painter.setPen(pen)
            painter.drawLine(rect.topLeft(), rect.topRight())
            painter.drawLine(rect.bottomLeft(), rect.bottomRight())

        if (row == self.current_row() and self.has_focus_or_context() and column in [C.COL_START]):
            pen = QPen()
            pen.setWidth(10)
            pen.setColor(QColor('#007041'))
            painter.setPen(pen)
            dyt = QPoint(0, 5)
            dyb = QPoint(0, 4)
            painter.drawLine(rect.bottomLeft() - dyb, rect.topLeft() + dyt)

    def sizeHint(self, style, model_index):
        """Override Qt method."""
        column = model_index.column()
        if column in [C.ACTION_COLUMNS, C.COL_PACKAGE_TYPE]:
            return QSize(32, 32)
        return QItemDelegate.sizeHint(self, style, model_index)


class TableCondaPackages(QTableView):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Conda packages table view."""

    WIDTH_TYPE = 24
    WIDTH_NAME = 120
    WIDTH_ACTIONS = 24
    WIDTH_VERSION = 90

    sig_status_updated = Signal(object, object, object, object)
    sig_conda_action_requested = Signal(str, int, str, object, object, object)
    sig_pip_action_requested = Signal(str, int)
    sig_actions_updated = Signal(int)
    sig_next_focus = Signal()
    sig_previous_focus = Signal()

    def __init__(self, parent):
        """Conda packages table view."""
        super().__init__(parent)
        self._parent = parent
        self._searchbox = ''
        self._filterbox = C.ALL
        self._delegate = CustomDelegate(self)
        self.row_count = None
        self._advanced_mode = True
        self._current_hover_row = None
        self._menu = None
        self._palette = {}

        # To manage icon states
        self._model_index_clicked = None
        self.valid = False
        self.column_ = None
        self.current_index = None

        # To prevent triggering the keyrelease after closing a dialog but hitting enter on it
        self.pressed_here = False

        self.source_model = None
        self.proxy_model = None

        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.verticalHeader().hide()
        self.setSortingEnabled(True)
        self.setMouseTracking(True)

        self._delegate.current_row = self.current_row
        self._delegate.current_hover_row = self.current_hover_row
        self._delegate.update_index = self.update
        self._delegate.has_focus_or_context = self.has_focus_or_context
        self.setItemDelegate(self._delegate)
        self.setShowGrid(False)
        self.setWordWrap(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Header setup
        self._hheader = self.horizontalHeader()
        # This should be handled by qtpy but is not working
        if PYQT5:
            self._hheader.setSectionResizeMode(QHeaderView.Fixed)
        elif PYQT4:
            try:
                self._hheader.setSectionResizeMode(QHeaderView.Fixed)
            except Exception:  # pylint: disable=broad-except
                self._hheader.setResizeMode(QHeaderView.Fixed)
        self._hheader.setStyleSheet("""QHeaderView {border: 0px; border-radius: 0px;};""")
        self.sortByColumn(C.COL_NAME, Qt.AscendingOrder)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.hide_columns()

    def setup_model(self, packages, data, metadata_links=None):
        """Setup model content."""
        self.proxy_model = MultiColumnSortFilterProxy(self)
        self.source_model = CondaPackagesModel(self, packages, data)
        self.proxy_model.setSourceModel(self.source_model)
        self.setModel(self.proxy_model)
        self.metadata_links = metadata_links if metadata_links else {}  # pylint: disable=attribute-defined-outside-init

        # NOTE: packages sizes... move to a better place?
        packages_sizes = {}
        for name in packages:
            packages_sizes[name] = packages[name].get('size')
        self._packages_sizes = packages_sizes  # pylint: disable=attribute-defined-outside-init

        # Custom Proxy Model setup
        self.proxy_model.setDynamicSortFilter(True)

        def filter_text(row, text, status):  # pylint: disable=unused-argument
            """Filter text helper function."""
            in_name = all(t in row[C.COL_NAME].lower() for t in to_text_string(text).lower().split())
            in_desc = all(t in row[C.COL_DESCRIPTION].lower() for t in to_text_string(text).split())
            return in_name or in_desc

        def filter_status(row, text, status):  # pylint: disable=unused-argument
            """Filter status helper function."""
            test1 = to_text_string(row[C.COL_STATUS]) in to_text_string(status)
            test2 = to_text_string(row[C.COL_ACTION]) in to_text_string(status)
            return test1 or test2

        self.model().add_filter_function('text-search', filter_text)
        self.model().add_filter_function('status-search', filter_status)

        # Signals and slots
        self.verticalScrollBar().valueChanged.connect(self.resize_rows)

        self.hide_columns()
        self.resize_rows()
        self.refresh_actions()

    def resize_rows(self):
        """Resize rows to fit the content."""
        delta_y = 10
        height = self.height()
        y = 0
        while y < height:
            row = self.rowAt(y)
            self.resizeRowToContents(row)
            row_height = self.rowHeight(row)
            self.setRowHeight(row, row_height + delta_y)
            y += self.rowHeight(row) + delta_y

    def hide_columns(self):
        """Hide unused columns."""
        for col in C.COLUMNS:
            self.showColumn(col)

        hide = HIDE_COLUMNS
        if self._advanced_mode:
            columns = C.ACTION_COLUMNS[:]
            columns.remove(C.COL_ACTION)
            hide += columns
        else:
            hide += [C.COL_ACTION]

        for col in hide:
            self.hideColumn(col)

    def filter_changed(self):  # pylint: disable=too-many-branches
        """Trigger the filter."""
        group = self._filterbox
        text = self._searchbox

        if group in [C.ALL]:
            group = '-'.join(
                [
                    to_text_string(C.INSTALLED),
                    to_text_string(C.UPGRADABLE),
                    to_text_string(C.NOT_INSTALLED),
                    to_text_string(C.DOWNGRADABLE),
                    to_text_string(C.MIXGRADABLE)
                ]
            )
        elif group in [C.INSTALLED]:
            group = '-'.join(
                [
                    to_text_string(C.INSTALLED),
                    to_text_string(C.UPGRADABLE),
                    to_text_string(C.DOWNGRADABLE),
                    to_text_string(C.MIXGRADABLE)
                ]
            )
        elif group in [C.UPGRADABLE]:
            group = '-'.join([to_text_string(C.UPGRADABLE), to_text_string(C.MIXGRADABLE)])
        elif group in [C.DOWNGRADABLE]:
            group = '-'.join([to_text_string(C.DOWNGRADABLE), to_text_string(C.MIXGRADABLE)])
        elif group in [C.SELECTED]:
            group = '-'.join(
                [
                    to_text_string(C.ACTION_INSTALL),
                    to_text_string(C.ACTION_REMOVE),
                    to_text_string(C.ACTION_UPGRADE),
                    to_text_string(C.ACTION_DOWNGRADE),
                    to_text_string(C.ACTION_UPDATE),
                ]
            )
        else:
            group = to_text_string(group)

        if self.proxy_model is not None:
            self.proxy_model.set_filter(text, group)
            self.resize_rows()

        # Update label count
        count = self.verticalHeader().count()
        if count == 0:
            count_text = _('0 packages available ')
        elif count == 1:
            count_text = _('1 package available ')
        elif count > 1:
            count_text = to_text_string(count) + _(' packages available ')

        if text != '':
            count_text = count_text + _('matching "{0}"').format(text)

        # Give information on selected packages
        selected_text = ''
        if self.source_model:
            action_count = self.source_model.get_action_count()
            if action_count:
                plural = 's' if action_count != 1 else ''
                selected_text = f'{action_count} package{plural} selected'

        self.sig_status_updated.emit(count_text, selected_text, None, None)

    def search_string_changed(self, text):
        """Update the search string text."""
        text = to_text_string(text)
        self._searchbox = text
        self.filter_changed()

    def filter_status_changed(self, text):
        """Update the type string selection."""
        self._filterbox = C.COMBOBOX_VALUES.get(text)
        self.filter_changed()

    def resizeEvent(self, event):
        """Override Qt method."""
        w = self.width()
        width_start = 8
        width_end = 0

        if self._advanced_mode:
            action_cols = [C.COL_ACTION]
        else:
            action_cols = [C.COL_UPGRADE, C.COL_INSTALL, C.COL_REMOVE, C.COL_DOWNGRADE]

        self.horizontalHeader().setMinimumSectionSize(
            min(self.WIDTH_TYPE, self.WIDTH_NAME, self.WIDTH_VERSION, self.WIDTH_ACTIONS),
        )

        self.setColumnWidth(C.COL_START, width_start)
        self.setColumnWidth(C.COL_PACKAGE_TYPE, self.WIDTH_TYPE)
        self.setColumnWidth(C.COL_NAME, self.WIDTH_NAME)
        self.setColumnWidth(C.COL_VERSION, self.WIDTH_VERSION)
        w_new = w - (
            width_start + self.WIDTH_ACTIONS + self.WIDTH_TYPE + self.WIDTH_NAME + self.WIDTH_VERSION +
            (len(action_cols)) * self.WIDTH_ACTIONS + width_end
        ) + 10
        self.setColumnWidth(C.COL_DESCRIPTION, w_new)
        self.setColumnWidth(C.COL_END, width_end)

        for col in action_cols:
            self.setColumnWidth(col, self.WIDTH_ACTIONS)
        QTableView.resizeEvent(self, event)
        self.resize_rows()

    def update_visible_rows(self):
        """Update range of visible rows close to selected row."""
        current_index = self.currentIndex()
        row = current_index.row()

        if self.proxy_model:
            for r in range(row - 50, row + 50):
                for co in C.COLUMNS:
                    index = self.proxy_model.index(r, co)
                    self.update(index)
            self.resize_rows()

    def current_row(self):
        """Return the currently selected row."""
        if self._menu and self._menu.isVisible():
            return self.currentIndex().row()
        if self.hasFocus():
            return self.currentIndex().row()
        return -1

    def current_hover_row(self):
        """Return the currently hovered row."""
        return self._current_hover_row

    def has_focus_or_context(self):
        """Return if the table has focus of if the context menu is on."""
        return self.hasFocus() or (self._menu and self._menu.isVisible())

    def mouseMoveEvent(self, event):
        """Override Qt method."""
        super().mouseMoveEvent(event)
        pos = event.pos()
        self._current_hover_row = self.rowAt(pos.y())

    def leaveEvent(self, event):
        """Override Qt method."""
        super().leaveEvent(event)
        self._current_hover_row = None
        self.repaint()

    def keyPressEvent(self, event):  # pylint: disable=too-many-branches
        """Override Qt method."""
        index = self.currentIndex()
        key = event.key()
        rows = self.verticalHeader().count()

        if MAC:
            if key == Qt.Key_Home:
                self.scrollToTop()
                self.setCurrentIndex(self.model().index(0, 0))
            elif key == Qt.Key_End:
                self.scrollToBottom()
                self.setCurrentIndex(self.model().index(rows - 1, 0))
            elif key == Qt.Key_Up:
                previous = index.row() - 1 if index.row() - 1 > 0 else 0
                self.setCurrentIndex(self.model().index(previous, 0))
            elif key == Qt.Key_Down:
                next_ = index.row() + 1 if index.row() + 1 < rows else rows - 1
                self.setCurrentIndex(self.model().index(next_, 0))

        if key in [Qt.Key_Enter, Qt.Key_Return]:
            # self.action_pressed(index)
            self.setCurrentIndex(self.proxy_model.index(index.row(), C.COL_ACTION))
            self.pressed_here = True
        elif key in [Qt.Key_Tab]:
            new_row = index.row() + 1
            if not self.proxy_model or new_row == self.proxy_model.rowCount():
                self.sig_next_focus.emit()
            else:
                new_index = self.proxy_model.index(new_row, 0)
                self.setCurrentIndex(new_index)
        elif key in [Qt.Key_Backtab]:
            new_row = index.row() - 1
            if new_row < 0:
                self.sig_previous_focus.emit()
            else:
                new_index = self.proxy_model.index(new_row, 0)
                self.setCurrentIndex(new_index)
        else:
            QTableView.keyPressEvent(self, event)

        self.update_visible_rows()

    def keyReleaseEvent(self, event):
        """Override Qt method."""
        QTableView.keyReleaseEvent(self, event)
        key = event.key()
        index = self.currentIndex()
        if key in [Qt.Key_Enter, Qt.Key_Return] and self.pressed_here:
            self.context_menu_requested(event)
        elif key in [Qt.Key_Menu]:
            self.setCurrentIndex(self.proxy_model.index(index.row(), C.COL_ACTION))
            self.context_menu_requested(event, right_click=True)
        self.pressed_here = False
        self.update_visible_rows()

    def mousePressEvent(self, event):
        """Override Qt method."""
        QTableView.mousePressEvent(self, event)

        pos = QPoint(event.x(), event.y())
        index = self.indexAt(pos)
        model = self.source_model

        if self.proxy_model is None or self.source_model is None:
            return

        model_index = self.proxy_model.mapToSource(index)
        row = model_index.row()

        if row == -1:
            return

        column = model_index.column()
        row_data = self.source_model.row(row)
        remove_actions = bool(self.source_model.count_remove_actions())
        install_actions = bool(self.source_model.count_install_actions())

        action = row_data[C.COL_ACTION]
        status = row_data[C.COL_STATUS]

        right_click = event.button() == Qt.RightButton
        left_click = event.button() == Qt.LeftButton

        if column == C.COL_ACTION:
            if right_click or (left_click and status != C.NOT_INSTALLED):
                self.context_menu_requested(event)
            elif left_click and status == C.NOT_INSTALLED:
                # 1-click install/uncheck if not installed
                if action == C.ACTION_NONE and not remove_actions:
                    self.set_action_status(model_index, C.ACTION_INSTALL)
                elif status:
                    self.set_action_status(model_index, C.ACTION_NONE)

        elif (
            column == C.COL_VERSION and model.is_upgradable(model_index) and left_click and not install_actions
            and not remove_actions
        ):
            # 1-click update
            self.set_action_status(model_index, C.ACTION_UPDATE)

        self.update_visible_rows()

    @staticmethod
    def _convert_action_index_to_name(action_index):
        index_action_name_map = {
            100: 'Unmark',
            101: 'Mark for installation',
            102: 'Mark for removal',
            103: 'Mark for update',
        }

        return index_action_name_map.get(action_index, 100)

    def mouseReleaseEvent(self, event):  # pylint: disable=unused-argument
        """Override Qt method."""
        self.update_visible_rows()

    def set_action_status(self, model_index, status=C.ACTION_NONE, version=None):
        """Set model index action status."""
        telemetry.ANALYTICS.instance.event('select-package', {'action': self._convert_action_index_to_name(status)})
        self.source_model.set_action_status(model_index, status, version)
        self.filter_changed()
        self.refresh_actions()

    def context_menu_requested(  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
            self, event, right_click=False,
    ):
        """Custom context menu."""
        if self.proxy_model is None:
            return

        self._menu = QMenu(self)
        left_click = not right_click
        index = self.currentIndex()
        model_index = self.proxy_model.mapToSource(index)
        row_data = self.source_model.row(model_index.row())
        column = model_index.column()
        name = row_data[C.COL_NAME]
        # package_type = row_data[C.COL_PACKAGE_TYPE]
        versions = self.source_model.get_package_versions(name)
        current_version = self.source_model.get_package_version(name)
        action_version = row_data[C.COL_ACTION_VERSION]
        package_status = row_data[C.COL_STATUS]
        package_type = row_data[C.COL_PACKAGE_TYPE]

        remove_actions = bool(self.source_model.count_remove_actions())
        install_actions = bool(self.source_model.count_install_actions())
        update_actions = bool(self.source_model.count_update_actions())

        if column in [C.COL_ACTION] and left_click:
            is_installable = self.source_model.is_installable(model_index)
            is_removable = self.source_model.is_removable(model_index)
            is_upgradable = self.source_model.is_upgradable(model_index)

            action_status = self.source_model.action_status(model_index)
            actions = []
            action_unmark = create_action(
                self,
                _('Unmark'),
                triggered=lambda: self.set_action_status(model_index, C.ACTION_NONE, current_version)
            )
            action_install = create_action(
                self, _('Mark for installation'), toggled=lambda: self.set_action_status(model_index, C.ACTION_INSTALL)
            )
            action_update = create_action(
                self, _('Mark for update'), toggled=lambda: self.set_action_status(model_index, C.ACTION_UPDATE, None)
            )
            action_remove = create_action(
                self,
                _('Mark for removal'),
                toggled=lambda: self.set_action_status(model_index, C.ACTION_REMOVE, current_version)
            )
            version_actions = []
            for version in reversed(versions):

                def trigger(model_index=model_index, action=C.ACTION_INSTALL, version=version):
                    return lambda: self.set_action_status(model_index, status=action, version=version)

                if version == current_version:
                    version_action = create_action(
                        self, version, icon=QIcon(), triggered=trigger(model_index, C.ACTION_INSTALL, version)
                    )
                    if not is_installable:
                        version_action.setCheckable(True)
                        version_action.setChecked(True)
                        version_action.setDisabled(True)
                elif version != current_version:
                    if (
                        (version in versions and versions.index(version)) >
                        (current_version in versions and versions.index(current_version))
                    ):
                        upgrade_or_downgrade_action = C.ACTION_UPGRADE
                    else:
                        upgrade_or_downgrade_action = C.ACTION_DOWNGRADE

                    if is_installable:
                        upgrade_or_downgrade_action = C.ACTION_INSTALL

                    version_action = create_action(
                        self,
                        version,
                        icon=QIcon(),
                        triggered=trigger(model_index, upgrade_or_downgrade_action, version)
                    )
                if action_version == version:
                    version_action.setCheckable(True)
                    version_action.setChecked(True)

                version_actions.append(version_action)

            install_versions_menu = QMenu('Mark for specific version installation', self)
            add_actions(install_versions_menu, version_actions)
            actions = [action_unmark, action_install, action_update, action_remove]
            actions += [None, install_versions_menu]

            # Disable firing of signals, while setting the checked status
            for ac in actions + version_actions:
                if ac:
                    ac.blockSignals(True)

            if action_status == C.ACTION_NONE:
                action_unmark.setEnabled(False)
                action_install.setEnabled(is_installable)
                action_update.setEnabled(is_upgradable)
                action_remove.setEnabled(is_removable)

                if install_actions:
                    # Invalidate remove and update if install actions selected
                    action_update.setDisabled(True)
                    action_remove.setDisabled(True)
                elif remove_actions:
                    # Invalidate install/update if remove actions already
                    action_install.setDisabled(True)
                    action_update.setDisabled(True)
                elif update_actions:
                    # Invalidate install/update if remove actions already
                    action_install.setDisabled(True)
                    action_remove.setDisabled(True)

                install_versions_menu.setDisabled(False)
            elif action_status == C.ACTION_INSTALL:
                action_unmark.setEnabled(True)
                action_install.setEnabled(False)
                action_install.setChecked(True)
                action_update.setEnabled(False)
                action_remove.setEnabled(False)
            elif action_status == C.ACTION_REMOVE:
                action_unmark.setEnabled(True)
                action_install.setEnabled(False)
                action_update.setEnabled(False)
                action_remove.setEnabled(False)
                action_remove.setChecked(True)
            elif action_status == C.ACTION_UPDATE:
                action_unmark.setEnabled(True)
                action_install.setEnabled(False)
                action_update.setEnabled(False)
                action_update.setChecked(True)
                action_remove.setEnabled(False)
            elif action_status in [C.ACTION_UPGRADE, C.ACTION_DOWNGRADE]:
                action_unmark.setEnabled(True)
                action_install.setEnabled(False)
                action_update.setEnabled(False)
                action_update.setChecked(False)
                action_remove.setEnabled(False)
                install_versions_menu.setEnabled(False)

            if package_status == C.NOT_INSTALLED:
                action_remove.setEnabled(False)
                action_update.setEnabled(False)

            if package_type == C.PIP_PACKAGE:
                action_unmark.setEnabled(False)
                action_install.setEnabled(False)
                action_update.setEnabled(False)
                action_remove.setEnabled(False)

            # Enable firing of signals, while setting the checked status
            for ac in actions + version_actions:
                if ac:
                    ac.blockSignals(False)

                install_versions_menu.setDisabled(True)

            install_versions_menu.setEnabled(len(version_actions) > 1 and not remove_actions and not update_actions)
        elif right_click:
            metadata = self.metadata_links.get(name, {})
            pypi = metadata.get('pypi', '')
            home = metadata.get('home', '')
            dev = metadata.get('dev', '')
            docs = metadata.get('docs', '')

            q_pypi = QIcon(get_image_path('python.png'))
            q_home = QIcon(get_image_path('home.png'))
            q_docs = QIcon(get_image_path('conda_docs.png'))

            if 'git' in dev:
                q_dev = QIcon(get_image_path('conda_github.png'))
            elif 'bitbucket' in dev:
                q_dev = QIcon(get_image_path('conda_bitbucket.png'))
            else:
                q_dev = QIcon()

            actions = []

            if pypi != '':
                actions.append(
                    create_action(self, _('Python Package Index'), icon=q_pypi, triggered=lambda: self.open_url(pypi))
                )
            if home != '':
                actions.append(create_action(self, _('Homepage'), icon=q_home, triggered=lambda: self.open_url(home)))
            if docs != '':
                actions.append(
                    create_action(self, _('Documentation'), icon=q_docs, triggered=lambda: self.open_url(docs))
                )
            if dev != '':
                actions.append(create_action(self, _('Development'), icon=q_dev, triggered=lambda: self.open_url(dev)))
        if actions and len(actions) > 1:
            # self._menu = QMenu(self)
            add_actions(self._menu, actions)

            if event.type() == QEvent.KeyRelease:
                rect = self.visualRect(index)
                global_pos = self.viewport().mapToGlobal(rect.bottomRight())
            else:
                pos = QPoint(event.x(), event.y())
                global_pos = self.viewport().mapToGlobal(pos)

            self._menu.popup(global_pos)

    def get_actions(self):
        """Return currently selected actions."""
        if self.source_model:
            return self.source_model.get_actions()
        return None

    def clear_actions(self):
        """Clear selected actions."""
        index = self.currentIndex()
        if self.source_model:
            self.source_model.clear_actions()
            self.refresh_actions()
            self.filter_changed()
        self.setFocus()
        self.setCurrentIndex(index)

    def refresh_actions(self):
        """Refresh package selected actions."""
        if self.source_model:
            actions_per_package_type = self.source_model.get_actions()
            number_of_actions = 0
            for _, actions in actions_per_package_type.items():
                for data in actions.values():
                    number_of_actions += len(data)
            self.sig_actions_updated.emit(number_of_actions)

    def update_style_sheet(self):
        """Update custom CSS style sheet."""

    @staticmethod
    def open_url(url):
        """
        Open link from action in default operating system browser.

        ADD TRACKING!.
        """
        if url:
            QDesktopServices.openUrl(QUrl(url))
            # NOTE: Add tracker
