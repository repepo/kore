# -*- coding: utf-8 -*-

# pylint: disable=attribute-defined-outside-init,no-name-in-module,unused-argument

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Conda channel selector dialog."""

from collections import defaultdict
import contextlib
from itertools import product
import os
import typing
from urllib import parse

from qtpy.QtCore import QEvent, QRegExp, QSize, Qt, Signal
from qtpy.QtGui import QKeySequence, QRegExpValidator
from qtpy.QtWidgets import (
    QAbstractItemView, QAction, QApplication, QFrame, QHBoxLayout, QListWidget, QListWidgetItem, QMenu, QProgressBar,
    QSizePolicy, QVBoxLayout,
)

from anaconda_navigator.api.anaconda_api import AnacondaAPI
from anaconda_navigator.config import CONF, AnacondaBrand
from anaconda_navigator.utils import get_domain_from_api_url
from anaconda_navigator.utils.misc import path_is_writable, convert_file_url_to_path
from anaconda_navigator.utils.styles import SASS_VARIABLES
from anaconda_navigator.widgets import (
    ButtonDanger, ButtonNormal, ButtonPrimary, FrameBase, LabelBase, LineEditBase, SpacerHorizontal, SpacerVertical,
)
from anaconda_navigator.widgets.dialogs import DialogBase


# --- Widgets used in CSS styling
# -----------------------------------------------------------------------------

class LabelConfigLocation(LabelBase):
    """Label displaying the configuration location."""


class LabelChannelInfo(LabelBase):
    """Label displaying channel information."""


class FrameChannels(FrameBase):
    """Frame used in CSS styling."""


class ButtonRemoveChannel(ButtonDanger):
    """Button that emits signal on focus."""

    sig_focused = Signal()

    def focusInEvent(self, event):
        """Override Qt method."""
        super().focusInEvent(event)
        self.sig_focused.emit()


# --- Heper widgets
# -----------------------------------------------------------------------------
class WorkerMock:  # pylint: disable=too-few-public-methods
    """Worker mock to handle `defaults` channel special case."""
    def __init__(self, item=None, url=None, repodata_url=None):
        """Worker mock to handle `defaults` channel special case."""
        self.item = item
        self.url = url
        self.repodata_url = url


class LineEditChannel(LineEditBase):
    """
    Custom line edit that uses different validators for text and url.

    More info:
    http://conda.pydata.org/docs/config.html#channel-locations-channels

    Valid entries:
    - defaults  <- Special case
    - <some-channel-name>
    - https://conda.anaconda.org/<channel>/<package>
    - https://conda.anaconda.org/t/<token>/<package>
    - http://<some.custom.url>/<channel>
    - https://<some.custom.url>/<channel>
    - file:///<some-local-directory>
    """

    VALID_RE = QRegExp(r'^[A-Za-z][A-Za-z0-9/_-]+$|^https?://.*|^file:///.*')  # pylint: disable=invalid-name

    sig_return_pressed = Signal()
    sig_escape_pressed = Signal()
    sig_copied = Signal()

    def __init__(self, *args, **kwargs):
        """Custom line edit that uses different validators for text and url."""
        super().__init__(*args, **kwargs)
        self._validator = QRegExpValidator(self.VALID_RE)
        self.menu = QMenu(parent=self)
        self.setValidator(self._validator)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def event(self, event):
        """Override Qt method."""
        if (event.type() == QEvent.MouseButtonPress and event.buttons() & Qt.RightButton and not self.isEnabled()):
            self.show_menu(event.pos())
            return True
        return super().event(event)

    def keyPressEvent(self, event):  # pylint: disable=invalid-name
        """Override Qt method."""
        key = event.key()

        # Display a copy menu in case the widget is disabled.
        if event.matches(QKeySequence.Paste):
            clipboard = QApplication.clipboard()
            text = clipboard.text()
            if self.VALID_RE.exactMatch(text):
                self.setText(text)
                return
        else:
            if key in [Qt.Key_Return, Qt.Key_Enter]:
                self.sig_return_pressed.emit()
            elif key in [Qt.Key_Escape]:
                self.sig_escape_pressed.emit()
        super().keyPressEvent(event)

    def show_menu(self, pos):
        """Show copy menu for channel item."""
        self.menu.clear()
        copy = QAction('&Copy', self.menu)
        copy.triggered.connect(self.copy_text)
        self.menu.addAction(copy)
        self.menu.setEnabled(True)
        self.menu.exec_(self.mapToGlobal(pos))

    def copy_text(self):
        """Copy channel text to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text())
        self.sig_copied.emit()


# --- Main list widgets
# -----------------------------------------------------------------------------
class ListWidgetItemChannel(QListWidgetItem):  # pylint: disable=too-many-instance-attributes
    """Conda channels list widget item used in CSS styling."""
    def __init__(self, channel=None, location=None):
        """Conda channels list widget item used in CSS styling."""
        super().__init__()
        self.channel = channel if channel else ''
        self.location = location if location else ''
        self.new_channel_help_text = 'Enter a channel name, file location, or url.'
        location_label = 'Path to configuration file: {location}'

        # Widgets
        self.widget = FrameChannels()
        self.new_channel_help = LabelConfigLocation(self.new_channel_help_text)
        self.new_channel_help.setVisible(False)
        self.label_location = LabelConfigLocation(location_label)
        self.label_location.setVisible(False)
        self.text_channel = LineEditChannel()
        self.label_info = LabelChannelInfo()
        self.button_remove = ButtonRemoveChannel()

        # Widgets setup
        self.button_remove.setVisible(path_is_writable(location))
        self.text_channel.setText(channel)

        # Layouts
        layout_name = QVBoxLayout()
        layout_name.addWidget(self.text_channel)
        layout_name.addWidget(self.new_channel_help)
        layout_name.addWidget(self.label_location)
        self.label_location.setToolTip(location_label)

        layout_frame = QHBoxLayout()
        layout_frame.addLayout(layout_name)
        layout_frame.addStretch()
        layout_frame.addWidget(self.label_info)
        layout_frame.addWidget(self.button_remove)
        self.widget.setLayout(layout_frame)
        self.setSizeHint(self.widget_size())

    def set_editable(self, value):
        """Set the editable status of the channel textbox."""
        self.text_channel.setEnabled(value)
        self.text_channel.setFocus()

    @staticmethod
    def widget_size():
        """Return the size defined in the SASS file."""
        return QSize(SASS_VARIABLES.WIDGET_CHANNEL_TOTAL_WIDTH, SASS_VARIABLES.WIDGET_CHANNEL_TOTAL_HEIGHT)


class ListWidgetChannels(QListWidget):
    """Conda channels list widget."""

    sig_channel_added = Signal(object)
    sig_channel_removed = Signal(object)
    sig_channel_status = Signal(bool)
    sig_channel_checked = Signal()
    sig_status_updated = Signal(object, object, object, object)
    sig_focus_fixed = Signal()

    def __init__(  # pylint: disable=too-many-arguments
        self,
        parent=None,
        api=None,
        main_url='https://anaconda.org',
        api_url='https://api.anaconda.org',
        conda_url='https://conda.anaconda.org'
    ):
        """Conda channels list widget."""
        super().__init__(parent)

        # Variables
        self._items = []
        self.api = api
        logged_brand: typing.Optional[str]
        logged_brand, logged_api_url = CONF.get_logged_data()

        if logged_api_url and logged_brand != AnacondaBrand.ANACONDA_ORG:
            main_url = get_domain_from_api_url(logged_api_url)
            api_url = parse.urljoin(main_url, 'api')
            conda_url = parse.urljoin(main_url, 'conda')

        self.api_url = api_url
        self.main_url = main_url
        self.conda_url = conda_url
        self.repeat_error = False

        # Widget setup
        self.setObjectName('ListWidgetChannels')
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setFrameStyle(QListWidget.Plain)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setViewMode(QListWidget.ListMode)
        self.setFocusPolicy(Qt.NoFocus)
        self.setUniformItemSizes(True)

    def addItem(self, item):  # pylint: disable=invalid-name
        """
        Override Qt method.

        Add a content item to the list.
        """
        super().addItem(item)
        self.setItemWidget(item, item.widget)

        item.text_channel.sig_return_pressed.connect(lambda: self.validate_channel(item))
        item.text_channel.sig_escape_pressed.connect(lambda: self.escape_pressed(item))
        item.button_remove.clicked.connect(lambda: self.remove_channel(item))
        item.button_remove.sig_focused.connect(lambda v=None: self.fix_focus(item))

        # Adding an empty channel triggers edit mode
        if not item.channel.strip().lower():
            self.is_editing = True
            self.scrollToBottom()
            item.set_editable(True)
            item.text_channel.textEdited.connect(self.check_repeat)
            item.button_remove.setVisible(False)
        else:
            self.is_editing = False
            self._items.append(item)

        item.text_channel.setToolTip(item.channel)
        item.label_info.setDisabled(True)

    def escape_pressed(self, item):
        """Handle cancellation of ongoing new channel editing."""
        self.takeItem(self.count() - 1)

    def fix_focus(self, item):
        """Set the current row based on focus of child widgets."""
        row = self._items.index(item)
        self.setCurrentRow(row)
        self.sig_focus_fixed.emit()

    def update_style_sheet(self):
        """Update custom CSS style sheet."""
        for item in self._items:
            with contextlib.suppress(BaseException):
                # This error is just in case the C++ object has been deleted and it is not crucial to log.
                item.setSizeHint(item.widget_size())
        self.update()
        self.repaint()

    def remove_channel(self, item):
        """Remove the selected channel."""
        if item in self._items:
            index = self._items.index(item)
            self.takeItem(index)
            self._items.remove(item)
            self.sig_channel_removed.emit(item.channel)

    def _channel_url_validated(self, worker, valid, error):
        """Callback for channel url validation."""
        self.setDisabled(False)
        item = worker.item
        if valid:
            item.set_editable(False)
            item.channel = worker.channel
            item.text_channel.setText(worker.channel)
            item.text_channel.setToolTip(worker.channel)
            item.new_channel_help.setVisible(False)
            item.label_location.setVisible(False)
            item.button_remove.setVisible(True)
            self.show_tool_tip(item)
            self._items.append(item)
            self.sig_channel_added.emit(worker.channel)
            self.sig_channel_status.emit(True)
            self.update_style_sheet()
            self.sig_status_updated.emit('', '', None, None)
        else:
            if CONF.get('main', 'offline_mode'):
                error_text, short_error_text = "Can't validate channel due to offline mode.", 'Offline Mode'
            else:
                error_text, short_error_text = (f'<b>{worker.url}</b> is not a valid conda channel.', 'Invalid channel')

            self.sig_status_updated.emit(short_error_text, '', None, None)
            item.set_editable(True)
            self.show_tool_tip(item, text=error_text)
            self.sig_channel_status.emit(False)

        self.sig_channel_checked.emit()

    @staticmethod
    def show_tool_tip(item, text=''):
        """Set the tooltip in case of errors."""
        widget = item.label_info
        widget.setToolTip(text)
        widget.setEnabled(bool(text))

    def check_repeat(self, text):
        """Check that given channel (text) and source is not in list."""
        compare = []
        for source, channels in self.sources.items():
            for channel in channels:
                lower_channel = channel.lower()
                compare.append((source, lower_channel))

        text = text.lower().strip()
        current_compare = (self.api._conda_api.user_rc_path, text)  # pylint: disable=protected-access
        self.repeat_error = current_compare in compare

    def validate_channel(self, item):
        """Validate entered channel with current api url."""
        channel = item.text_channel.text().strip()
        self.sig_status_updated.emit('Validating channel...', '', 0, 0)
        item.channel = channel

        if self.repeat_error:
            # Channel is already in list
            self.show_tool_tip(item, text=f'Channel <b>{channel}</b> is in list already.')
            worker = WorkerMock(item=item, url='', repodata_url='')
            worker.channel = channel
            self._channel_url_validated(worker, False, None)
            return None

        if not channel:
            # Inform user channel is empty!
            self.show_tool_tip(item, text='Channel cannot be empty.')
            self.sig_status_updated.emit('Channel cannot be empty', '', None, None)
            worker = WorkerMock(item=item, url='', repodata_url='')
            worker.channel = channel
            self._channel_url_validated(worker, False, None)
            return None

        # Try to validate
        if channel == 'defaults':
            worker = WorkerMock(item=item, url=channel, repodata_url=channel)
            worker.channel = channel
            return self._channel_url_validated(worker, True, None)

        if channel.startswith('file:///'):
            # Its a local folder, only check that the folder exists
            worker = WorkerMock(item=item, url=channel, repodata_url=channel)
            worker.channel = channel
            path = convert_file_url_to_path(channel)
            valid = os.path.isdir(path)
            return self._channel_url_validated(worker, valid, None)

        if channel.startswith(('https://', 'http://')):
            anaconda_api_url = CONF.get('main', 'anaconda_api_url')

            anaconda_org_api_netloc = parse.urlparse(anaconda_api_url).netloc
            anaconda_org_netloc = anaconda_org_api_netloc.split('api.')[-1]
            url = channel
            if parse.urlparse(channel).netloc in (anaconda_org_api_netloc, anaconda_org_netloc):
                # User entered https://anaconda.org/<channel> or
                # User entered https://anaconda.org/t/<token>/<package>
                url = channel.replace(self.main_url, self.conda_url)
                channel = url
        else:
            token = CONF.get('main', 'anaconda_server_token')
            conda_url = self.conda_url if self.conda_url.endswith('/') else self.conda_url + '/'
            url = parse.urljoin(conda_url, channel)

            logged_brand: typing.Optional[str]
            logged_brand, _ = CONF.get_logged_data()

            if logged_brand == AnacondaBrand.TEAM_EDITION:
                url = parse.urljoin(self.main_url, f'api/repo/t/{token}/{channel}')

        url = url[:-1] if url[-1] == '/' else url
        repodata_url = f'{url}/noarch/repodata.json'
        worker = self.api.download_is_valid_url(repodata_url)
        worker.sig_finished.connect(self._channel_url_validated)
        worker.item = item
        worker.url = url
        worker.channel = channel
        worker.repodata_url = repodata_url
        self.setDisabled(True)
        self.show_tool_tip(item)
        return None

    @property
    def sources(self):
        """Return the channels."""
        config_sources = defaultdict(list)

        for item in self._items:
            config_sources[item.location].append(item.channel)

        return config_sources


class DialogChannels(DialogBase):  # pylint: disable=too-many-instance-attributes
    """Dialog to add delete and select active conda package channels."""

    sig_channels_updated = Signal(object, object)  # added, removed
    sig_setup_ready = Signal()
    sig_check_ready = Signal()
    WIDTH = 550  # pylint: disable=invalid-name

    def __init__(self, parent=None):  # pylint: disable=too-many-statements
        """Dialog to add delete and select active conda pacakge channels ."""
        super().__init__(parent)
        self._parent = parent
        self._conda_url = 'https://conda.anaconda.org'
        self.api = AnacondaAPI()
        self.initial_sources = None
        self.config_sources = {}

        # Widgets
        self.list = ListWidgetChannels(parent=self, api=self.api)
        self.label_info = LabelBase('Manage channels you want Navigator to include.')
        self.label_status = LabelBase('Collecting sources...')
        self.progress_bar = QProgressBar(self)
        self.button_add = ButtonNormal('Add...')
        self.button_cancel = ButtonNormal('Cancel')
        self.button_ok = ButtonPrimary('Update channels')

        # Widget setup
        self.frame_title_bar.setVisible(False)
        self.list.setFrameStyle(QFrame.NoFrame)
        self.list.setFrameShape(QFrame.NoFrame)
        self.setWindowFlags(self.windowFlags() | Qt.Popup)
        self.setWindowOpacity(0.96)
        self.setMinimumHeight(300)
        self.setMinimumWidth(self.WIDTH)
        self.setModal(True)

        # Layout
        layout_button = QHBoxLayout()
        layout_button.addWidget(self.label_info)
        layout_button.addStretch()
        layout_button.addWidget(self.button_add)

        layout_ok = QHBoxLayout()
        layout_ok.addWidget(self.label_status)
        layout_ok.addWidget(SpacerHorizontal())
        layout_ok.addWidget(self.progress_bar)
        layout_ok.addWidget(SpacerHorizontal())
        layout_ok.addStretch()
        layout_ok.addWidget(self.button_cancel)
        layout_ok.addWidget(SpacerHorizontal())
        layout_ok.addWidget(self.button_ok)

        layout = QVBoxLayout()
        layout.addLayout(layout_button)
        layout.addWidget(SpacerVertical())
        layout.addWidget(self.list)
        layout.addWidget(SpacerVertical())
        layout.addWidget(SpacerVertical())
        layout.addLayout(layout_ok)
        self.setLayout(layout)

        # Signals
        self.button_add.clicked.connect(self.add_channel)
        self.button_ok.clicked.connect(self.update_channels)
        self.button_cancel.clicked.connect(self.reject)
        self.list.sig_status_updated.connect(self.update_status)
        self.list.sig_channel_added.connect(lambda v=None: self.set_tab_order())
        self.list.sig_channel_added.connect(lambda v=None: self.button_ok.setFocus())
        self.list.sig_channel_removed.connect(lambda v=None: self.set_tab_order())
        self.list.sig_channel_removed.connect(lambda v=None: self.button_ok.setFocus())
        self.list.sig_channel_checked.connect(self.sig_check_ready)
        self.list.sig_channel_status.connect(self.refresh)

        self.button_add.setDisabled(True)
        self.button_ok.setDisabled(True)
        self.button_cancel.setDisabled(False)
        self.update_status(action='Collecting sources...', value=0, max_value=0)

    @staticmethod
    def _group_sources_and_channels(sources):
        """
        Flatten sources and channels dictionary to list of tuples.

        [(source, channel), (source, channel)...]
        """
        grouped = []
        for source, channels in sources.items():
            grouped.extend(product([source], channels))
        return grouped

    def keyPressEvent(self, event):  # pylint: disable=invalid-name
        """Override Qt method."""
        key = event.key()
        if key in [Qt.Key_Escape]:
            if self.list.is_editing:
                self.refresh()
                self.list.is_editing = False
            else:
                self.reject()

    # --- Public API
    # -------------------------------------------------------------------------
    def update_style_sheet(self):
        """Update custom css style sheets."""
        self.setMinimumWidth(SASS_VARIABLES.WIDGET_CHANNEL_DIALOG_WIDTH)

        with contextlib.suppress(BaseException):
            self.list.update_style_sheet()

    def setup(self, worker, conda_config_data, error):
        """Setup the channels widget."""
        self.config_sources = conda_config_data.get('config_sources', [])
        self.button_add.setDisabled(False)

        for source, data in self.config_sources.items():
            channels = data.get('channels', [])
            for channel in channels:
                item = ListWidgetItemChannel(channel=channel, location=source)
                item.set_editable(False)
                item.new_channel_help.setVisible(False)
                item.label_location.setVisible(False)
                self.list.addItem(item)

        self.set_tab_order()
        self.button_add.setFocus()
        self.button_ok.setDefault(True)
        self.button_cancel.setEnabled(True)

        self.initial_sources = self.list.sources.copy()
        self.update_status()
        self.sig_setup_ready.emit()

    def set_tab_order(self):
        """Fix the tab ordering in the list."""
        if self.list._items:  # pylint: disable=protected-access
            self.setTabOrder(self.button_add, self.list._items[0].button_remove)  # pylint: disable=protected-access
            self.setTabOrder(self.list._items[-1].button_remove, self.button_cancel)  # pylint: disable=protected-access

        self.setTabOrder(self.button_cancel, self.button_ok)
        self.refresh()

    def add_channel(self):
        """Add new conda channel."""
        user_rc_path = self.api._conda_api.user_rc_path  # pylint: disable=protected-access
        item = ListWidgetItemChannel(channel='', location=user_rc_path)
        item.new_channel_help.setVisible(True)
        item.label_location.setVisible(True)
        self.list.addItem(item)
        self.refresh(False)

    def update_channels(self):
        """Update channels list and status."""
        sources = self.list.sources

        original = self._group_sources_and_channels(self.initial_sources)
        updated = self._group_sources_and_channels(sources)

        if sorted(original) != sorted(updated):
            self.sig_channels_updated.emit(*self.sources)
            self.accept()
        else:
            self.reject()

    def refresh(self, channel_status=True):
        """Update enable/disable status based on item count."""
        self.button_add.setEnabled(channel_status and bool(self.list.count))
        self.button_ok.setEnabled(channel_status)
        self.button_cancel.setEnabled(True)

        if self.list.count() == 0:
            self.button_add.setEnabled(True)
            self.button_ok.setEnabled(False)

    def update_status(self, action='', message='', value=None, max_value=None):
        """Update the status and progress bar of the widget."""
        visible = bool(action)
        self.label_status.setText(action)
        self.label_status.setVisible(visible)
        if value is not None and max_value is not None:
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, max_value)
            self.progress_bar.setValue(value)
        else:
            self.progress_bar.setVisible(False)

    @property
    def sources(self):
        """Return sources to add and remove from config."""
        original = self._group_sources_and_channels(self.initial_sources)
        updated = self._group_sources_and_channels(self.list.sources)

        original = set(original)
        updated = set(updated)

        add = updated - original
        remove = original - updated

        return add, remove
