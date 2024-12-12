# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-module-docstring

from __future__ import annotations

__all__ = ()

from qtpy import QtCore
from qtpy.QtWidgets import QTableView, QHeaderView  # pylint: disable=no-name-in-module
from qtpy import QtGui


class SelectableChannelsItem(QtGui.QStandardItem):  # pylint: disable=too-few-public-methods
    """
    Standard table item object representation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setEditable(False)

    def __lt__(self, other):
        if self.isCheckable():
            return not self.checkState() in (other.checkState(), QtCore.Qt.Checked)
        return super().__lt__(other)


class SelectableChannelsListTable(QTableView):
    """
    Table object representation for selecting channels while doing login to Anaconda Server instance.
    """
    NAME_INDEX = 0
    MIRROR_INDEX = 1
    PRIVACY_INDEX = 2
    OWNERS_INDEX = 3
    DEFAULT_CHANNELS_INDEX = 4
    CHANNELS_INDEX = 5

    TABLE_HEADERS = ['Name', 'Mirror', 'Privacy', 'Owners', 'Add to default_channels', 'Add to channels']

    def __init__(self, parent, table_data, channels=None, default_channels=None):
        super().__init__()

        if not channels:
            channels = []
        if not default_channels:
            default_channels = []

        self.parent = parent
        self.apply_data_model(table_data, channels, default_channels)

        for index in range(len(self.TABLE_HEADERS)):
            self.horizontalHeader().setSectionResizeMode(index, QHeaderView.ResizeToContents)

        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(False)
        self.setSortingEnabled(True)

        self.horizontalHeader().resizeSection(self.CHANNELS_INDEX, 120)

    def apply_data_model(self, table_data, channels, default_channels):
        """
        Creates the model data for the table depending on passed data in 'table_data' attribute.

        :param dict[str, str/int] table_data: The data which will be hold in model.
        :param list[str] channels: List of 'channels' names which must be checked by default.
        :param list[str] default_channels: List of 'default_channels' which must be checked by default.
        """
        model = QtGui.QStandardItemModel()
        model.setHorizontalHeaderLabels(self.TABLE_HEADERS)

        for row, channel in enumerate(table_data):
            name: str = '/'.join(filter(bool, (channel.get('parent', None), channel['name'])))

            model.setItem(row, self.NAME_INDEX, SelectableChannelsItem(name))
            model.setItem(row, self.MIRROR_INDEX, self.generate_checkable_item(bool(channel['mirror_count']), False))
            model.setItem(row, self.PRIVACY_INDEX, SelectableChannelsItem(channel['privacy']))
            model.setItem(row, self.OWNERS_INDEX, SelectableChannelsItem(', '.join(filter(bool, channel['owners']))))
            model.setItem(row, self.DEFAULT_CHANNELS_INDEX, self.generate_checkable_item(name in default_channels))
            model.setItem(row, self.CHANNELS_INDEX, self.generate_checkable_item(name in channels))

        self.setModel(model)
        self.sortByColumn(self.MIRROR_INDEX, QtCore.Qt.DescendingOrder)

    @staticmethod
    def generate_checkable_item(is_checked=False, is_enabled=True):
        """
        Creates a table item with a checkbox and empty string.

        :param bool is_checked: Defines if the checkbox must be checked.
        :param bool is_enabled: Defines if the checkbox must be enabled.

        :return SelectableChannelsItem: The created table item object.
        """
        checkable_item = SelectableChannelsItem()
        checkable_item.setCheckable(True)
        checkable_item.setEnabled(is_enabled)
        checkable_item.setTextAlignment(QtCore.Qt.AlignCenter)

        if is_checked:
            checkable_item.setCheckState(QtCore.Qt.Checked)

        return checkable_item

    def get_selected_channels(self):
        """
        Returns the lists with 'default_channels' and 'channels' names fetched from the model,
        where on row 4 and 5 (default_channels and channels) checkboxes are in the checked state.

        :return tuple[list[str], list[str]]
        """
        default_channels = []
        channels = []

        for row in range(self.model().rowCount()):
            data_index = self.model().index(row, self.NAME_INDEX)

            if self.model().item(row, self.DEFAULT_CHANNELS_INDEX).checkState() == QtCore.Qt.Checked:
                default_channels.append(self.model().data(data_index))

            if self.model().item(row, self.CHANNELS_INDEX).checkState() == QtCore.Qt.Checked:
                channels.append(self.model().data(data_index))

        return default_channels, channels
