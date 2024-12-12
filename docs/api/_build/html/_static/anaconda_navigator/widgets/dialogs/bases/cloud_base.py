# -*- coding: utf-8 -*-

"""Base for Cloud-styled dialogs."""

from __future__ import annotations

__all__ = ['CloudDialogBase']

import typing

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from anaconda_navigator.static import images
from anaconda_navigator.utils import singletons


CLOSE_DIALOG_ICON: typing.Final[singletons.Singleton[QtGui.QIcon]] = singletons.SingleInstanceOf(
    QtGui.QIcon,
    images.CLOSE_DIALOG_ICON_PATH,
)


class CloudDialogFrame(QtWidgets.QFrame):  # pylint: disable=too-few-public-methods
    """Primary frame of :class:`~CloudDialogBase`."""


class CloudDialogTobBorder(QtWidgets.QFrame):  # pylint: disable=too-few-public-methods
    """Green border of :class:`~CloudDialogBase`."""


class CloudDialogTitleFrame(QtWidgets.QFrame):  # pylint: disable=too-few-public-methods
    """Container for title bar of :class:`~CloudDialogBase`."""


class CloudDialogTitleLabel(QtWidgets.QLabel):  # pylint: disable=too-few-public-methods
    """Title to show in :class:`~CloudDialogBase`."""


class CloudDialogCloseButton(QtWidgets.QPushButton):  # pylint: disable=too-few-public-methods
    """Close button in :class:`~CloudDialogBase`."""


class CloudDialogBodyFrame(QtWidgets.QFrame):  # pylint: disable=too-few-public-methods
    """Container for user content in :class:`~CloudDialogBase`."""


class CloudDialogBase(QtWidgets.QDialog):  # pylint: disable=too-many-instance-attributes
    """Common base for dialogs in Cloud style."""

    def __init__(self, parent: typing.Optional[QtCore.QObject] = None) -> None:
        """Initialize new :class:`~CloudDialogBase` instance."""
        super().__init__(parent)

        self.setMinimumHeight(480)
        self.setMinimumWidth(640)

        self._top_border: typing.Final[CloudDialogTobBorder] = CloudDialogTobBorder()

        self._title_label: typing.Final[CloudDialogTitleLabel] = CloudDialogTitleLabel()

        self._close_dialog_button: typing.Final[CloudDialogCloseButton] = CloudDialogCloseButton()
        self._close_dialog_button.setFixedSize(QtCore.QSize(24, 24))
        self._close_dialog_button.setIcon(CLOSE_DIALOG_ICON.instance)
        self._close_dialog_button.setIconSize(QtCore.QSize(24, 24))
        self._close_dialog_button.clicked.connect(lambda: self.reject())  # pylint: disable=unnecessary-lambda

        self._title_frame_layout: typing.Final[QtWidgets.QHBoxLayout] = QtWidgets.QHBoxLayout()
        self._title_frame_layout.setAlignment(QtCore.Qt.AlignCenter)
        self._title_frame_layout.setContentsMargins(8, 0, 8, 0)
        self._title_frame_layout.setSpacing(0)
        self._title_frame_layout.addWidget(self._title_label)
        self._title_frame_layout.addStretch(1)
        self._title_frame_layout.addWidget(self._close_dialog_button)

        self._title_frame: typing.Final[CloudDialogTitleFrame] = CloudDialogTitleFrame()
        self._title_frame.setLayout(self._title_frame_layout)
        self._title_frame.setContentsMargins(0, 0, 0, 0)

        self._body_frame: typing.Final[CloudDialogBodyFrame] = CloudDialogBodyFrame()

        self._dialog_frame_layout: typing.Final[QtWidgets.QVBoxLayout] = QtWidgets.QVBoxLayout()
        self._dialog_frame_layout.setContentsMargins(0, 0, 0, 0)
        self._dialog_frame_layout.setSpacing(0)
        self._dialog_frame_layout.addWidget(self._top_border)
        self._dialog_frame_layout.addWidget(self._title_frame)
        self._dialog_frame_layout.addWidget(self._body_frame)

        self._dialog_frame: typing.Final[CloudDialogFrame] = CloudDialogFrame()
        self._dialog_frame.setLayout(self._dialog_frame_layout)

        self._dialog_layout: typing.Final[QtWidgets.QVBoxLayout] = QtWidgets.QVBoxLayout()
        self._dialog_layout.setAlignment(QtCore.Qt.AlignTop)
        self._dialog_layout.setContentsMargins(0, 0, 0, 0)
        self._dialog_layout.setSpacing(10)
        self._dialog_layout.addWidget(self._dialog_frame)

        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setSizeGripEnabled(False)
        self.setWindowFlags(QtCore.Qt.Dialog | QtCore.Qt.FramelessWindowHint)
        super().setLayout(self._dialog_layout)

    def layout(self) -> QtWidgets.QLayout:
        """Retrieve current dialog layout."""
        return self._body_frame.layout()

    def setWindowTitle(self, title: str) -> None:  # pylint: disable=invalid-name
        """Change title of the dialog window."""
        super().setWindowTitle(title)
        self._title_label.setText(title)

    def setLayout(self, layout: QtWidgets.QLayout) -> None:  # pylint: disable=invalid-name
        """Set content of the dialog."""
        self._body_frame.setLayout(layout)
