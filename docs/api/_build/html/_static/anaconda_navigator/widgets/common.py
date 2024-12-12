# -*- coding: utf-8 -*-

# pylint: disable=too-few-public-methods

"""Common widgets and their bases."""

from __future__ import annotations

__all__ = ['IconButton']

import typing

from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets

from anaconda_navigator.static import images
from anaconda_navigator import widgets
from anaconda_navigator.utils import singletons


class IconButton(QtWidgets.QPushButton):
    """Common control for icon-only buttons."""


class WarningIcon(QtWidgets.QLabel):
    """
    Icon with exclamation sign.

    Used by :class:`~WarningLabel` and its derivatives.
    """


class ErrorIcon(QtWidgets.QLabel):
    """
    Icon with exclamation sign in circle.
    """


class InfoIcon(QtWidgets.QLabel):
    """
    Icon with exclamation sign in circle.
    """


class LabelWithIcon(QtWidgets.QFrame):
    """
    Combined label with icon at the beginning.

    Icon is shown only when the main text is not empty.
    """

    sig_button_clicked = QtCore.Signal()

    def __init__(  # pylint: disable=too-many-arguments
            self,
            icon: typing.Optional[widgets.QLabel] = None,
            text: str = '',
            tooltip: str = '',
            button: str = '',
            alignment: typing.Any = QtCore.Qt.AlignTop,
    ) -> None:
        """Initialize new :class:`~WarningLabel` instance."""
        super().__init__()

        self.__icon: typing.Final[widgets.QLabel] = icon

        self.__content: typing.Final[widgets.LabelBase] = widgets.LabelBase()
        self.__content.setToolTip(tooltip)
        self.__content.setWordWrap(True)

        self.__button: typing.Final[widgets.LabelBase] = widgets.LabelBase()
        self.__button.setText(button)
        self.__button.sig_clicked.connect(self.sig_button_clicked)

        self.__layout: typing.Final[QtWidgets.QHBoxLayout] = QtWidgets.QHBoxLayout()
        self.__layout.addWidget(self.__icon, 0, alignment)
        self.__layout.addWidget(self.__content, 1, alignment)
        if button:
            self.__layout.addWidget(self.__button, 0, alignment)
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(4)

        self.setLayout(self.__layout)

        self.text = text

    @property
    def text(self) -> str:  # noqa: D401
        """Content of the main label."""
        return self.__content.text()

    @text.setter
    def text(self, value: str) -> None:
        """Update `text` value."""
        self.__content.setText(value)
        self.__icon.setVisible(bool(value))


class WarningLabel(LabelWithIcon):
    """
    Combined label with warning icon at the beginning.

    Icon is shown only when the main text is not empty.
    """

    def __init__(
            self,
            text: str = '',
            tooltip: str = '',
            button: str = '',
            alignment: typing.Any = QtCore.Qt.AlignTop,
    ) -> None:
        """Initialize new :class:`~WarningLabel` instance."""
        super().__init__(icon=WarningIcon(), text=text, tooltip=tooltip, button=button, alignment=alignment)


class WarningBlock(WarningLabel):
    """Same as :class:`~WarningLabel`, but with colored background."""

    def __init__(
            self,
            text: str = '',
            tooltip: str = '',
            button: str = '',
            alignment: typing.Any = QtCore.Qt.AlignTop,
    ) -> None:
        """Initialize new :class:`~WarningBlock` instance."""
        super().__init__(text=text, tooltip=tooltip, button=button, alignment=alignment)


class ErrorLabel(LabelWithIcon):
    """
    Combined label with error icon at the beginning.

    Icon is shown only when the main text is not empty.
    """

    def __init__(
            self,
            text: str = '',
            tooltip: str = '',
            button: str = '',
            alignment: typing.Any = QtCore.Qt.AlignTop,
    ) -> None:
        """Initialize new :class:`~ErrorLabel` instance."""
        super().__init__(icon=ErrorIcon(), text=text, tooltip=tooltip, button=button, alignment=alignment)


class ErrorBlock(ErrorLabel):
    """Same as :class:`~ErrorLabel`, but with colored background."""

    def __init__(
            self,
            text: str = '',
            tooltip: str = '',
            button: str = '',
            alignment: typing.Any = QtCore.Qt.AlignTop,
    ) -> None:
        """Initialize new :class:`~ErrorBlock` instance."""
        super().__init__(text=text, tooltip=tooltip, button=button, alignment=alignment)


class InfoLabel(LabelWithIcon):
    """
    Combined label with info icon at the beginning.

    Icon is shown only when the main text is not empty.
    """

    def __init__(
            self,
            text: str = '',
            tooltip: str = '',
            button: str = '',
            alignment: typing.Any = QtCore.Qt.AlignTop,
    ) -> None:
        """Initialize new :class:`~InfoLabel` instance."""
        super().__init__(icon=InfoIcon(), text=text, tooltip=tooltip, button=button, alignment=alignment)


class InfoBlock(InfoLabel):
    """Same as :class:`~InfoLabel`, but with colored background."""

    def __init__(
            self,
            text: str = '',
            tooltip: str = '',
            button: str = '',
            alignment: typing.Any = QtCore.Qt.AlignTop,
    ) -> None:
        """Initialize new :class:`~InfoBlock` instance."""
        super().__init__(text=text, tooltip=tooltip, button=button, alignment=alignment)


class TeamEditionServerAlert(QtWidgets.QWidget):
    """Show combined labels with icon about connectivity problems to TE Server."""

    def __init__(self) -> None:
        super().__init__()
        self.error_label: typing.Final[ErrorBlock] = ErrorBlock(
            text=(
                '<b>Anaconda Server cannot be reached.</b> '
                'Some actions may not work as expected until connection is restored.'
            ),
            alignment=QtCore.Qt.AlignVCenter,
        )
        self.info_label: typing.Final[InfoBlock] = InfoBlock(
            text='<b>Connection to Anaconda Anaconda Server server restored.</b>',
            button=f'<img src="{images.BLOCK_CLOSE_ICON_PATH}" height="16" width="16">',
            alignment=QtCore.Qt.AlignVCenter,
        )
        self.__timer: typing.Final[QtCore.QTimer] = QtCore.QTimer()

        layout: typing.Final[QtWidgets.QVBoxLayout] = QtWidgets.QVBoxLayout()
        layout.addWidget(self.error_label)
        layout.addWidget(self.info_label)

        self.info_label.sig_button_clicked.connect(self.hide_all)

        self.setLayout(layout)
        self.hide_all()

    def show_info(self, limit: typing.Optional[int] = 5000) -> None:
        """Show info label for `limit` seconds"""
        if not self.error_label.isVisible():
            self.error_label.hide()
            return

        self.show()
        self.error_label.hide()
        self.info_label.show()
        if limit:
            self.__timer.singleShot(limit, self.hide_all)

    def show_error(self) -> None:
        """Show error label"""
        self.show()
        self.error_label.show()
        self.info_label.hide()

    def hide_all(self) -> None:
        """Hide all labels."""
        self.hide()
        self.error_label.hide()
        self.info_label.hide()


PASSWORD_HIDDEN: typing.Final[singletons.Singleton[QtGui.QIcon]] = singletons.SingleInstanceOf(
    QtGui.QIcon,
    images.PASSWORD_HIDDEN,
)
PASSWORD_VISIBLE: typing.Final[singletons.Singleton[QtGui.QIcon]] = singletons.SingleInstanceOf(
    QtGui.QIcon,
    images.PASSWORD_VISIBLE,
)


class LineEdit(QtWidgets.QLineEdit):
    """Custom QLineEdit with additional toggles for password entry."""

    def __init__(self, *args: typing.Any, show_toggle: bool = False) -> None:
        """Initialize new :class:`~LineEdit` instance."""
        super().__init__(*args)

        toggle_action: typing.Optional[QtWidgets.QAction] = None
        if show_toggle:
            toggle_action = self.addAction(PASSWORD_HIDDEN.instance, QtWidgets.QLineEdit.TrailingPosition)
            toggle_action.triggered.connect(lambda _: self.setEchoMode())
        self._toggle_action: typing.Final[typing.Optional[QtWidgets.QAction]] = toggle_action

        if show_toggle:
            self.setEchoMode(QtWidgets.QLineEdit.Password)

    def setEchoMode(self, mode: typing.Optional[int] = None) -> None:  # pylint: disable=invalid-name
        """Set how text should be echoed to the user."""
        if mode is None:
            if self.echoMode() == QtWidgets.QLineEdit.Password:
                mode = QtWidgets.QLineEdit.Normal
            else:
                mode = QtWidgets.QLineEdit.Password
        super().setEchoMode(mode)

        if self._toggle_action is None:
            return
        if mode == QtWidgets.QLineEdit.Password:
            self._toggle_action.setIcon(PASSWORD_HIDDEN.instance)
        else:
            self._toggle_action.setIcon(PASSWORD_VISIBLE.instance)
