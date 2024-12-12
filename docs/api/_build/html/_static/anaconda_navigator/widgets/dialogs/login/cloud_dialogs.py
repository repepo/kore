# -*- coding: utf-8 -*-

"""Anaconda Cloud advertisement and sign-in dialogs."""

from __future__ import annotations

__all__ = ['CloudLoginPage']

import collections.abc
import os
import typing
import json
import urllib.parse

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from anaconda_navigator import config
from anaconda_navigator.config import preferences
from anaconda_navigator.api import cloud
from anaconda_navigator.utils import attribution
from anaconda_navigator.utils import download_manager
from anaconda_navigator.utils import telemetry
from anaconda_navigator.utils import url_utils
from anaconda_navigator.utils import workers


METADATA_CACHE_PATH: typing.Final[str] = os.path.join(config.CLOUD_CACHE, 'metadata.json')


class Page(typing.TypedDict):
    """Single advertisement page for a :class:`CloudLoginPage` dialog."""

    background: str


class Metadata(typing.TypedDict):
    """Definition of a metadata with advertisement definition."""

    pages: collections.abc.Sequence[Page]


def parse_metadata(download: download_manager.Download) -> Metadata | None:
    """Parse metadata for :class:`CloudLoginPage`."""
    try:
        stream: typing.TextIO
        with download.open('rt', encoding='utf-8') as stream:
            return json.load(stream)
    except (OSError, TypeError, ValueError):
        return None


class CloudLoginSingInButton(QtWidgets.QPushButton):
    """Style for 'Sign In' button."""


class CloudLoginCloseButton(QtWidgets.QPushButton):
    """Style for dialog close button."""


class BackgroundControlButton(QtWidgets.QPushButton):
    """Style for 'Sign In' button."""


class CloudLoginPage(QtWidgets.QDialog):  # pylint: disable=too-many-instance-attributes
    """Sign-in page for Cloud auth."""

    __image_cache__: typing.ClassVar[dict[str, QtGui.QBrush]] = {}
    __metadata__: typing.ClassVar[Metadata | None] = None

    def __init__(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Initialize new instance of a :class:`~CloudLoginPage`."""
        super().__init__(*args, **kwargs)

        self.__api: typing.Final[cloud._CloudAPI] = cloud.CloudAPI()

        self.current_page: int = -1

        self.dots_layout: typing.Final[QtWidgets.QHBoxLayout] = QtWidgets.QHBoxLayout()
        self.dots_layout.setAlignment(QtCore.Qt.AlignHCenter)
        self.dots_layout.setSpacing(24)

        close_button: typing.Final[CloudLoginCloseButton] = CloudLoginCloseButton('X', self)
        close_button.setCursor(QtCore.Qt.PointingHandCursor)
        close_button.clicked.connect(self.reject)

        self.top_bar_layout: typing.Final[QtWidgets.QHBoxLayout] = QtWidgets.QHBoxLayout()
        self.top_bar_layout.setAlignment(QtCore.Qt.AlignRight)
        self.top_bar_layout.setContentsMargins(0, 12, 16, 0)
        self.top_bar_layout.addWidget(close_button)

        self.sign_in_button: typing.Final[CloudLoginSingInButton] = CloudLoginSingInButton('Sign In Now')
        self.sign_in_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.sign_in_button.clicked.connect(self._login)

        sign_up_label: typing.Final[QtWidgets.QLabel] = QtWidgets.QLabel(
            'Don\'t have an account yet? <a href="#" style="color:#43b049; text-decoration:none">Sign Up</a>',
            self
        )
        sign_up_label.linkActivated.connect(self._create_account)
        sign_up_label.setAlignment(QtCore.Qt.AlignCenter)

        controls_layout: typing.Final[QtWidgets.QVBoxLayout] = QtWidgets.QVBoxLayout()
        controls_layout.addWidget(self.sign_in_button, alignment=QtCore.Qt.AlignHCenter)
        controls_layout.addWidget(sign_up_label, alignment=QtCore.Qt.AlignHCenter)
        controls_layout.setSpacing(24)
        controls_layout.addLayout(self.dots_layout)

        layout: typing.Final[QtWidgets.QVBoxLayout] = QtWidgets.QVBoxLayout(self)
        layout.addLayout(self.top_bar_layout)
        layout.addStretch()
        layout.addLayout(controls_layout)
        layout.setContentsMargins(0, 0, 0, 24)

        self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.FramelessWindowHint)

        self.sign_in_timer: typing.Final[QtCore.QTimer] = QtCore.QTimer(self)
        self.sign_in_timer.setInterval(5_000)
        self.sign_in_timer.setSingleShot(True)
        self.sign_in_timer.timeout.connect(self.__allow_sign_in)

        self.slideshow_timer: typing.Final[QtCore.QTimer] = QtCore.QTimer(self)
        self.slideshow_timer.setInterval(10_000)
        self.slideshow_timer.setSingleShot(True)
        self.slideshow_timer.timeout.connect(self.__next_page)

        pending: bool = self.__metadata__ is None
        self.__apply_metadata(self.__metadata__)
        if preferences.CLOUD_METADATA_SOURCE and pending:
            self.__download_metadata(preferences.CLOUD_METADATA_SOURCE)

        self.__login_thread: workers.TaskThread | None = None

        self.finished.connect(self._cancel_login)

    @staticmethod
    def _create_account() -> None:
        """React to selecting 'Sign Up' link."""
        url = attribution.POOL.settings.inject_url_parameters(
            'https://id.anaconda.cloud/ui/registration',
            utm_medium='connect-cloud',
            utm_content='signup',
        )
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
        telemetry.ANALYTICS.instance.event('redirect', {'url': str(url)})

    def _login(self) -> None:
        """React to selecting 'Sign In' option."""
        self.sign_in_button.setDisabled(True)
        self.sign_in_timer.start()

        self._cancel_login()

        worker: workers.TaskWorker = self.__api.login.worker()  # pylint: disable=no-member
        worker.signals.sig_done.connect(self._login_done)
        self.__login_thread = worker.thread()

    def _cancel_login(self) -> None:
        """Cancel login process."""
        thread: workers.TaskThread | None = self.__login_thread
        if thread is None:
            return

        try:
            thread.cancel()
        except TypeError:  # task already finished
            pass
        if not thread.wait(5_000):
            thread.terminate()

        self.__login_thread = None

    def _login_done(self, result: workers.TaskResult) -> None:
        """Process Cloud API response to login request."""
        if not self.isVisible():
            return

        QtWidgets.QApplication.restoreOverrideCursor()

        if result.status == workers.TaskStatus.CANCELED:
            return

        if result.status == workers.TaskStatus.SUCCEEDED:
            self.accept()
            return

        self.reject()

    def __allow_sign_in(self) -> None:
        """Allow user to click on the sign-in button."""
        self.sign_in_button.setEnabled(True)

    def __download_metadata(self, url: str) -> None:
        """Start download of .cloud dialog metadata."""
        download: download_manager.Download
        download = download_manager.Download(url).into(METADATA_CACHE_PATH).process(parse_metadata)
        download.sig_succeeded.connect(self.__download_backgrounds)
        download_manager.manager().execute(download)

    def __download_backgrounds(self, download: download_manager.Download) -> None:
        """Download backgrounds from the metadata."""
        metadata: Metadata | None = download.result
        if not metadata:
            return

        files: dict[str, str] = {}
        queue: list[download_manager.Download] = []

        page: Page
        for page in metadata['pages']:
            origin: str = urllib.parse.urljoin(download._url, page['background'])  # pylint: disable=protected-access
            try:
                page['background'] = files[origin]
            except KeyError:
                target: str = os.path.join(config.CLOUD_CACHE, url_utils.safe_file_name(origin))
                queue.append(download_manager.Download(origin).into(target).keep(download_manager.Keep.FRESH))
                page['background'] = files[origin] = target

        if queue:
            group: download_manager.Group = download_manager.Group(*queue).require(download_manager.Require.ALL)
            group.sig_succeeded.connect(lambda _: self.__apply_metadata(metadata))
            download_manager.manager().execute(group)

    def __apply_metadata(self, metadata: Metadata | None = None) -> None:
        """Update page background according to the metadata."""
        if metadata is None:
            metadata = preferences.CLOUD_DEFAULT_METADATA

        # prepare images for all pages
        pages: list[Page] = []
        for page in metadata.get('pages', []):
            if page['background'] not in self.__image_cache__:
                try:
                    self.__image_cache__[page['background']] = QtGui.QBrush(QtGui.QPixmap(page['background']))
                except Exception:  # nosec=B112 # pylint: disable=broad-exception-caught
                    continue
            pages.append(page)

        # skip processing metadata with no usable pages
        if not pages:
            return

        # store last valid metadata (will be reused for repeating dialog executions)
        metadata['pages'] = pages
        type(self).__metadata__ = metadata

        # determine the count of dots to show
        index: int = len(pages)
        if index <= 1:
            index = 0

        # remove excessive dots
        layout_item: QtWidgets.QLayoutItem | None
        while layout_item := self.dots_layout.takeAt(index):
            layout_item.widget().deleteLater()

        # add missing dots
        for index in range(self.dots_layout.count(), index):
            self.dots_layout.addWidget(self.__create_dot(index))

        # update background and selected dot
        self.switch_page(index=min(max(self.current_page, 0), len(pages) - 1))

    def __create_dot(self, index: int) -> BackgroundControlButton:
        """Create new dot control."""
        control: BackgroundControlButton = BackgroundControlButton()
        control.setCursor(QtCore.Qt.PointingHandCursor)
        control.clicked.connect(lambda _: self.switch_page(index=index))
        control.setCheckable(True)
        return control

    def switch_page(self, index: int) -> None:
        """Switch login page to the :code:`index`."""
        if not self.__metadata__:
            return
        try:
            page: Page = self.__metadata__['pages'][index]  # pylint: disable=unsubscriptable-object
        except IndexError:
            return

        next_item: QtWidgets.QLayoutItem | None
        previous_item: QtWidgets.QLayoutItem | None
        if next_item := self.dots_layout.itemAt(index):
            if previous_item := self.dots_layout.itemAt(self.current_page):
                previous_item.widget().setChecked(False)
            next_item.widget().setChecked(True)
            self.current_page = index

        palette: QtGui.QPalette = self.palette()
        palette.setBrush(QtGui.QPalette.Window, self.__image_cache__[page['background']])
        self.setPalette(palette)

        self.slideshow_timer.start()

    def __next_page(self) -> None:
        """Switch page to the next one by the :attr:`~CloudLoginPage.slideshow_timer`."""
        if self.__metadata__ is None:
            return

        index: int = self.current_page + 1
        if index >= len(self.__metadata__['pages']):  # pylint: disable=unsubscriptable-object
            index = 0

        self.switch_page(index)
