# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Dialogs for choosing login options."""

__all__ = ['AccountStatus', 'AccountState', 'AccountOutcome', 'AccountValue', 'AccountsDialog']

import enum
import html
import typing
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from anaconda_navigator import widgets
from anaconda_navigator import config


class AccountStateMapping(typing.TypedDict, total=False):
    """Dictionary with states for each available account."""

    cloud: 'AccountState'
    individual: 'AccountState'
    commercial: 'AccountState'
    team: 'AccountState'
    enterprise: 'AccountState'


class AccountStatus(enum.Enum):
    """Options for a state of a single accounts."""

    AVAILABLE = enum.auto()  # user can log in
    UNAVAILABLE = enum.auto()  # user can not log in
    ACTIVE = enum.auto()  # user is already logged in


class AccountState(typing.NamedTuple):  # pylint: disable=missing-class-docstring

    status: AccountStatus = AccountStatus.UNAVAILABLE
    username: typing.Optional[str] = None


class AccountOutcome(enum.Enum):
    """Options for dialog result."""

    REJECT = enum.auto()
    LOGIN_REQUEST = enum.auto()
    LOGOUT_REQUEST = enum.auto()


class AccountValue(enum.Enum):
    """Options for dialog value, that can be selected in the dialog."""

    CLOUD = enum.auto()
    INDIVIDUAL_EDITION = enum.auto()
    COMMERCIAL_EDITION = enum.auto()
    TEAM_EDITION = enum.auto()
    ENTERPRISE_EDITION = enum.auto()


class AccountsFrame(widgets.FrameBase):
    """Main frame for account selector dialog content."""

    def __init__(self, parent: typing.Optional[QtWidgets.QWidget] = None) -> None:
        """Initialize new :class:`~AccountsFrame` instance."""
        super().__init__(parent=parent)

        self.__layout: typing.Final[QtWidgets.QVBoxLayout] = QtWidgets.QVBoxLayout()
        self.__layout.setContentsMargins(0, 0, 0, 0)
        self.__layout.setSpacing(0)

        self.setLayout(self.__layout)

    def add_widget(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Add new widget to the internal layout."""
        self.__layout.addWidget(*args, **kwargs)


class AccountsLabel(widgets.FrameBase):
    """Common interface for label widgets."""

    def __init__(
            self,
            alignment: int,
            parent: typing.Optional[QtWidgets.QWidget] = None,
    ) -> None:
        """Initialize new :class:`~AccountsLabel` instance."""
        super().__init__(parent=parent)

        self.__content: typing.Final[QtWidgets.QLabel] = widgets.LabelBase()

        layout: typing.Final[QtWidgets.QVBoxLayout] = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.__content, 0, alignment | QtCore.Qt.AlignVCenter)

        self.setLayout(layout)

    @property
    def text(self) -> str:  # noqa: D401
        """Text to show."""
        return self.__content.text()

    @text.setter
    def text(self, value: str) -> None:
        """Update `text` value."""
        self.__content.setText(value)


class AccountsTitle(AccountsLabel):
    """Title for the dialog."""

    def __init__(self, parent: typing.Optional[QtWidgets.QWidget] = None) -> None:
        """Initialize new :class:`~AccountsTitle` instance."""
        super().__init__(alignment=QtCore.Qt.AlignHCenter, parent=parent)


class AccountsHeader(AccountsLabel):
    """Header for a single section in account selector dialog."""

    def __init__(self, parent: typing.Optional[QtWidgets.QWidget] = None) -> None:
        """Initialize new :class:`~AccountsHeader` instance."""
        super().__init__(alignment=QtCore.Qt.AlignLeft, parent=parent)


class AccountEntryPrefix(widgets.LabelBase):
    """Text to draw on the top of the account card."""


class AccountEntryTitle(widgets.LabelBase):
    """Label of an account title."""


class AccountEntryAction(widgets.LabelBase):
    """Place to put actions for an account (sign in, sign out, etc.)."""


class AccountEntryPostfix(widgets.LabelBase):
    """Text to draw on the bottom of the account card."""


class AccountEntry(widgets.FrameBase):  # pylint: disable=too-many-instance-attributes
    """Card for a single account."""

    sig_link_opened = QtCore.Signal(str)

    def __init__(self, parent: typing.Optional[QtWidgets.QWidget] = None) -> None:
        """Initialize new :class:`~AccountEntry` instance."""
        super().__init__(parent=parent)

        self.__prefix: typing.Final[AccountEntryPrefix] = AccountEntryPrefix()
        self.__prefix.linkActivated.connect(self.__open_link)

        self.__title: typing.Final[AccountEntryTitle] = AccountEntryTitle()
        self.__title.linkActivated.connect(self.__open_link)

        self.__action: typing.Final[AccountEntryAction] = AccountEntryAction()
        self.__action.linkActivated.connect(self.__open_link)

        self.__postfix: typing.Final[AccountEntryPostfix] = AccountEntryPostfix()
        self.__postfix.linkActivated.connect(self.__open_link)

        content_layout: typing.Final[QtWidgets.QHBoxLayout] = QtWidgets.QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self.__title, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        content_layout.addWidget(self.__action, 0, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        layout: typing.Final[QtWidgets.QVBoxLayout] = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.__prefix, 0, QtCore.Qt.AlignLeft)
        layout.addLayout(content_layout, 0)
        layout.addWidget(self.__postfix, 0, QtCore.Qt.AlignLeft)

        self.setLayout(layout)

    @classmethod
    def prepare(
            cls,
            title: str,
            alias: str,
            state: AccountState,
            open_link: typing.Optional[typing.Callable[[str], None]] = None,
    ) -> 'AccountEntry':
        """Create new :class:`~AccountEntry` instance with initial values."""
        result: AccountEntry = cls()
        result.prefix = ''
        result.title = html.escape(title)
        if state.status == AccountStatus.ACTIVE:
            result.action = (
                f'<a'
                f' href="navigator://{html.escape(alias)}/logout"'
                f' style="color: #0075A9; text-decoration: none">'
                f'SIGN OUT'
                f'</a>'
            )
            result.enabled = True
            if state.username:
                result.postfix = f'connected as {state.username}'
            else:
                result.postfix = 'connected'
        elif state.status == AccountStatus.AVAILABLE:
            result.action = (
                f'<a'
                f' href="navigator://{html.escape(alias)}/login"'
                f' style="color: #0075A9; text-decoration: none">'
                f'SIGN IN'
                f'</a>'
            )
            result.enabled = True
            result.postfix = ''
        else:
            result.action = 'SIGN IN'
            result.enabled = False
            result.postfix = ''
        if open_link is not None:
            result.sig_link_opened.connect(open_link)
        return result

    @property
    def action(self) -> str:  # noqa: D401
        """Content of the action part of the card."""
        return self.__action.text()

    @action.setter
    def action(self, value: str) -> None:
        """Update `action` value."""
        self.__action.setText(value)

    @property
    def enabled(self) -> bool:  # noqa: D401
        """Can user interact with this card."""
        return self.__title.isEnabled()

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Update `enabled` value."""
        self.__action.setEnabled(value)
        self.__postfix.setEnabled(value)
        self.__prefix.setEnabled(value)
        self.__title.setEnabled(value)

    @property
    def postfix(self) -> str:  # noqa: D401
        """Content of the postfix part of the card."""
        return self.__postfix.text()

    @postfix.setter
    def postfix(self, value: str) -> None:
        """Update `postfix` value."""
        self.__postfix.setText(value)

    @property
    def prefix(self) -> str:  # noqa: D401
        """Content of the prefix part of the card."""
        return self.__prefix.text()

    @prefix.setter
    def prefix(self, value: str) -> None:
        """Update `prefix` value."""
        self.__prefix.setText(value)

    @property
    def title(self) -> str:  # noqa: D401
        """Content of the title part of the card."""
        return self.__title.text()

    @title.setter
    def title(self, value: str) -> None:
        """Update `title` value."""
        self.__title.setText(value)

    def __open_link(self, link: str) -> None:
        """Process link opening action."""
        if link == '#':
            return
        self.sig_link_opened.emit(link)


class AccountsDialog(QtWidgets.QDialog):
    """
    Dialog for account management.

    :param parent: Parent window to associate this dialog with.
    :param anchor: Control in the `parent`, below which new :class:`~AccountsDialog` must be placed.
    :param states: Details about current account login states.
    """

    sig_accepted = QtCore.Signal(AccountOutcome, AccountValue)

    def __init__(
            self,
            parent: QtWidgets.QWidget,
            anchor: QtWidgets.QWidget,
            states: 'AccountStateMapping',
    ) -> None:
        """Initialize new :class:`~AccountsDialog` instance."""
        super().__init__(parent=parent)

        self.__outcome: AccountOutcome = AccountOutcome.REJECT
        self.__value: typing.Optional[AccountValue] = None

        self.__anchor: typing.Final[QtWidgets.QWidget] = anchor

        title: typing.Final[AccountsTitle] = AccountsTitle()
        title.text = 'Connect to Anaconda'

        cloud_account: typing.Final[AccountEntry] = AccountEntry.prepare(
            title='Anaconda Cloud',
            alias='cloud',
            state=states.get('cloud', AccountState()),
            open_link=self.__open_link,
        )

        repositories_header: typing.Final[AccountsHeader] = AccountsHeader()
        repositories_header.text = 'REPOSITORIES'

        individual_account: typing.Final[AccountEntry] = AccountEntry.prepare(
            title=config.AnacondaBrand.ANACONDA_ORG,
            alias='individual',
            state=states.get('individual', AccountState()),
            open_link=self.__open_link,
        )
        commercial_account: typing.Final[AccountEntry] = AccountEntry.prepare(
            title=config.AnacondaBrand.COMMERCIAL_EDITION,
            alias='commercial',
            state=states.get('commercial', AccountState()),
            open_link=self.__open_link,
        )
        team_account: typing.Final[AccountEntry] = AccountEntry.prepare(
            title=config.AnacondaBrand.TEAM_EDITION,
            alias='team',
            state=states.get('team', AccountState()),
            open_link=self.__open_link,
        )
        enterprise_account: typing.Final[AccountEntry] = AccountEntry.prepare(
            title=config.AnacondaBrand.ENTERPRISE_EDITION,
            alias='enterprise',
            state=states.get('enterprise', AccountState()),
            open_link=self.__open_link,
        )

        frame = AccountsFrame()
        frame.add_widget(title)
        frame.add_widget(cloud_account)
        frame.add_widget(repositories_header)
        frame.add_widget(individual_account)
        frame.add_widget(commercial_account)
        frame.add_widget(team_account)
        frame.add_widget(enterprise_account)

        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(frame)

        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setLayout(layout)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint | QtCore.Qt.Popup)

    @property
    def outcome(self) -> AccountOutcome:  # noqa: D401
        """Result of dialog execution."""
        return self.__outcome

    @property
    def value(self) -> AccountValue:  # noqa: D401
        """Selected option in the dialog."""
        if self.__value is None:
            raise AttributeError('value is not available')
        return self.__value

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # pylint: disable=invalid-name
        """
        Process `show` event of the dialog.

        This is used to place :class:`~AccountsDialog` to the expected location.
        """
        super().showEvent(event)
        self.move(self.__anchor.mapToGlobal(QtCore.QPoint(
            self.__anchor.width() + 14 - self.width(),
            self.__anchor.height() + 10,
        )))

    def __open_link(self, link: str) -> None:
        """Process link opening action."""
        self.__outcome, self.__value = {
            'navigator://cloud/login': (AccountOutcome.LOGIN_REQUEST, AccountValue.CLOUD),
            'navigator://cloud/logout': (AccountOutcome.LOGOUT_REQUEST, AccountValue.CLOUD),
            'navigator://individual/login': (AccountOutcome.LOGIN_REQUEST, AccountValue.INDIVIDUAL_EDITION),
            'navigator://individual/logout': (AccountOutcome.LOGOUT_REQUEST, AccountValue.INDIVIDUAL_EDITION),
            'navigator://commercial/login': (AccountOutcome.LOGIN_REQUEST, AccountValue.COMMERCIAL_EDITION),
            'navigator://commercial/logout': (AccountOutcome.LOGOUT_REQUEST, AccountValue.COMMERCIAL_EDITION),
            'navigator://team/login': (AccountOutcome.LOGIN_REQUEST, AccountValue.TEAM_EDITION),
            'navigator://team/logout': (AccountOutcome.LOGOUT_REQUEST, AccountValue.TEAM_EDITION),
            'navigator://enterprise/login': (AccountOutcome.LOGIN_REQUEST, AccountValue.ENTERPRISE_EDITION),
            'navigator://enterprise/logout': (AccountOutcome.LOGOUT_REQUEST, AccountValue.ENTERPRISE_EDITION),
        }[link]
        self.accept()
        self.sig_accepted.emit(self.outcome, self.value)
