# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Dialogs for choosing correct account for further environment action."""

__all__ = ['SelectorOutcome', 'SelectorValue', 'SelectorDialog']

import enum
import html
import typing
from qtpy import QtCore
from qtpy import QtWidgets
import requests
from anaconda_navigator import widgets
from anaconda_navigator.api import cloud
from anaconda_navigator.widgets import dialogs
from anaconda_navigator.widgets import common as global_common
from .. import utilities
from . import common

if typing.TYPE_CHECKING:
    from anaconda_navigator.api.cloud.tools import error_parsers as cloud_error_parsers


T = typing.TypeVar('T')
T_co = typing.TypeVar('T_co', covariant=True)

HEADING_ERROR_TEMPLATE: typing.Final[str] = '<span style="font-size: 14px; font-weight: 500">{content}</span>'
FOOTER_ERROR_TEMPLATE: typing.Final[str] = (
    '<span style="color: #808080; font-size: 13px; font-style: italic; font-weight: 300">{content}</span>'
)


class SelectorOutcome(enum.IntEnum):
    """Options for selector dialog result."""

    ACCEPT = enum.auto()
    REJECT = enum.auto()
    LOGIN_REQUEST = enum.auto()


class SelectorValue(enum.IntEnum):
    """Options for target values, that can be selected from selector dialogs."""

    LOCAL = enum.auto()
    CLOUD = enum.auto()


class SelectorOption(QtCore.QObject):
    """Container for single available option, to select in :class:`~SelectorDialog`."""

    def __init__(
            self,
            name: str,
            parent: typing.Optional[QtCore.QObject] = None,
    ) -> None:
        """Initialize new :class:`~SelectorOption` instance."""
        super().__init__(parent=parent)

        self.__radio: typing.Final[widgets.RadioButtonBase] = widgets.RadioButtonBase()

        self.__label: typing.Final[QtWidgets.QLabel] = widgets.LabelBase()
        self.__label.setText(f'<span style="font-size: 14px; font-weight: 700">{html.escape(name)}</span>')
        self.__label.sig_clicked.connect(self.__label_clicked)

    @property
    def radio(self) -> widgets.RadioButtonBase:  # noqa: D401
        """Radio control of the option."""
        return self.__radio

    @property
    def label(self) -> QtWidgets.QLabel:  # noqa: D401
        """Label control of the option."""
        return self.__label

    @property
    def checked(self) -> bool:  # noqa: D401
        """Current option is selected."""
        return self.__radio.isChecked()

    @checked.setter
    def checked(self, value: bool) -> None:  # noqa: D401
        """Update `checked` value."""
        self.__radio.setChecked(value)
        self.__radio.setFocus()

    @property
    def enabled(self) -> bool:  # noqa: D401
        """Current option can be selected."""
        return self.__radio.isEnabled()

    @enabled.setter
    def enabled(self, value: bool) -> None:  # noqa: D401
        """Update `enabled` value."""
        self.__radio.setEnabled(value)
        self.__label.setEnabled(value)

    def __label_clicked(self) -> None:
        """Process clicking on the label."""
        self.checked = True


class ProgressFrame(QtWidgets.QFrame):  # pylint: disable=too-few-public-methods
    """Frame with progress bar, that might be easily hidden without breaking a layout."""

    def __init__(self) -> None:
        """Initialize new :class:`~ProgressFrame` instance."""
        super().__init__()

        self.__progress_bar: typing.Final[QtWidgets.QProgressBar] = QtWidgets.QProgressBar()
        self.__progress_bar.setRange(0, 0)
        self.__progress_bar.setVisible(False)

        progress_layout: typing.Final[QtWidgets.QHBoxLayout] = QtWidgets.QHBoxLayout()
        progress_layout.addWidget(self.__progress_bar, 1, QtCore.Qt.AlignVCenter)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(0)

        self.setLayout(progress_layout)
        self.setContentsMargins(0, 0, 0, 0)

    @property
    def progress_bar(self) -> QtWidgets.QProgressBar:  # noqa: D401
        """Wrapped progress bar control."""
        return self.__progress_bar


class MappingProxy(typing.Generic[T, T_co], typing.Mapping[T, T_co]):
    """Read-only proxy for mappings."""

    __slots__ = ('__source',)

    def __init__(self, source: typing.Mapping[T, T_co]) -> None:
        """Initialize new :class:`~MappingProxy` instance."""
        self.__source: typing.Final[typing.Mapping[T, T_co]] = source

    def __getitem__(self, key: T) -> T_co:
        """Retrieve item by it's key."""
        return self.__source[key]

    def __len__(self) -> int:
        """Retrieve total number of records in collection."""
        return len(self.__source)

    def __iter__(self) -> typing.Iterator[T]:
        """Iterate through item keys."""
        return iter(self.__source)


class SelectorForm(QtCore.QObject):
    """Group of :class:`~SelectorOption` instances."""

    sig_value_changed = QtCore.Signal(SelectorValue)

    def __init__(self, parent: typing.Optional[QtCore.QObject] = None) -> None:
        """Initialize new :class:`~SelectorForm` instance."""
        super().__init__(parent=parent)

        self.__options: typing.Final[typing.Dict[SelectorValue, SelectorOption]] = {}

        self.__layout: typing.Final[QtWidgets.QGridLayout] = QtWidgets.QGridLayout()
        self.__layout.setColumnMinimumWidth(0, 42)
        self.__layout.setColumnStretch(1, 1)

        self.__row: int = 0
        self.__need_spacer: bool = False

        self.__group: typing.Final[QtWidgets.QButtonGroup] = QtWidgets.QButtonGroup()
        self.__group.setParent(parent)
        self.__group.setExclusive(True)

        self.__value: typing.Optional[SelectorValue] = None

    @property
    def layout(self) -> QtWidgets.QGridLayout:  # noqa: D401
        """UI element with all options."""
        return self.__layout

    @property
    def options(self) -> typing.Mapping[SelectorValue, SelectorOption]:  # noqa: D401
        """
        Collection of added selector options.

        This collection is read-only. If you want to add new items - use :meth:`~SelectorForm.__setitem__`.
        """
        return MappingProxy(source=self.__options)

    @property
    def value(self) -> SelectorValue:  # noqa: D401
        """Current selected option."""
        if self.__value is None:
            raise AttributeError('value must be set before accessing')
        return self.__value

    @value.setter
    def value(self, value: SelectorValue) -> None:  # noqa: D401
        """Select value from added options."""
        option: typing.Optional[SelectorOption] = self.__options.get(value, None)
        if option is None:
            raise ValueError(f'option for {value.name!r} is not set')
        if not option.enabled:
            raise ValueError(f'option for {value.name!r} is not enabled')
        option.checked = True

    def __add_anything(self, *args: typing.Any, stay_on_row: bool = False) -> typing.Sequence[typing.Any]:
        """
        Common method, which prepares arguments to use with `addWidget` or `addLayout` methods of the `layout`.

        :param args: Original arguments for the `add*` methods.

                     Row must be skipped, as it is added automatically with this method.
        :param stay_on_row: Do not move to the next row after adding current element.
        """
        arguments: typing.List[typing.Any] = list(args)
        if len(arguments) < 1:
            raise TypeError('at least one argument must be provided')
        arguments.insert(1, self.__row)
        if len(arguments) == 2:
            arguments.append(0)

        if not stay_on_row:
            self.__row += 1

        return arguments

    def add_layout(self, *args: typing.Any, stay_on_row: bool = False) -> None:
        """
        Add new child layout to current layout.

        .. warning::

            Do not provide `row` value!

            New items always adds to the end of the layout.

            If you want to add multiple elements on a single row - set `stay_on_row` to notify that next element should
            also be placed on the current row.
        """
        self.__layout.addLayout(*self.__add_anything(*args, stay_on_row=stay_on_row))

    def add_spacer(self) -> None:
        """Add new spacer at the end of the layout."""
        self.__layout.addWidget(widgets.SpacerVertical(), self.__row, 0, 1, 2, QtCore.Qt.AlignCenter)
        self.__need_spacer = False
        self.__row += 1

    def add_widget(self, *args: typing.Any, stay_on_row: bool = False) -> None:
        """
        Add new widget to current layout.

        .. warning::

            Do not provide `row` value!

            New items always adds to the end of the layout.

            If you want to add multiple elements on a single row - set `stay_on_row` to notify that next element should
            also be placed on the current row.
        """
        self.__layout.addWidget(*self.__add_anything(*args, stay_on_row=stay_on_row))

    def __getitem__(self, value: SelectorValue) -> SelectorOption:
        """Retrieve :class:`~SelectorOption` by its value."""
        return self.__options[value]

    def __setitem__(self, value: SelectorValue, option: SelectorOption) -> None:
        """Add new :class:`~SelectorOption` to the group."""
        if value in self.__options:
            raise KeyError(f'option is already set for {value.name!r}')
        self.__options[value] = option

        if self.__need_spacer:
            self.add_spacer()

        self.__layout.addWidget(option.radio, self.__row, 0, 1, 1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.__layout.addWidget(option.label, self.__row, 1, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.__need_spacer = True
        self.__row += 1

        self.__group.addButton(option.radio)

        def register_change(checked: bool) -> None:
            if checked:
                self.__value = value
                self.sig_value_changed.emit(value)

        option.radio.toggled.connect(register_change)


class SelectorRecord(typing.NamedTuple):
    """
    Single account entry in :class:`~SelectorDialog`.

    E.g. "Local" or "Cloud"
    """

    value: SelectorValue
    title: str


class SelectorDialogControls:
    """
    Collection of all controls in :class:`~SelectorDialog`.

    Used for locking parts of dialog on account option change or progress.
    """

    __slots__ = ('all', 'account_local', 'account_cloud', '__suspended')

    def __init__(self) -> None:
        """Initialize new :class:`~SelectorDialogControl` instance."""
        self.all: utilities.WidgetGroup = utilities.WidgetGroup()
        self.account_local: utilities.WidgetGroup = utilities.WidgetGroup()
        self.account_cloud: utilities.WidgetGroup = utilities.WidgetGroup()
        self.__suspended: utilities.WidgetGroup = utilities.WidgetGroup()

    def restore(self) -> None:
        """Restore all controls, that were suspended previously."""
        self.__suspended.enable()
        self.__suspended = utilities.WidgetGroup()

    def suspend(self) -> None:
        """Suspend all currently enabled controls."""
        delta: utilities.WidgetGroup = self.all.only_enabled()
        delta.disable()
        self.__suspended += delta


class SelectorDialog(dialogs.DialogBase):
    """
    Dialog for selecting target where to backup environment.

    This dialog is split into four parts, each with its own initialization method:

    - header (:code:`__init_header__`) with dialog caption
    - form (:code:`__init_form__`) with account options (local, Cloud)
    - footer (:code:`__init__footer__`) with additional controls after the form
    - actions (:code:`__init_actions__`) with common action controls (accept button, cancel button, progress bar)
    """

    def __init__(self, parent: typing.Optional[QtWidgets.QWidget] = None) -> None:
        """Initialize new :class:`~BackupSelectorDialog` instance."""
        super().__init__(parent=parent)

        self.__controls: typing.Final[SelectorDialogControls] = SelectorDialogControls()

        self.__outcome: SelectorOutcome = SelectorOutcome.REJECT
        self.__value: typing.Optional[SelectorValue] = None

        # Controls

        self.__form: typing.Final[SelectorForm] = SelectorForm(parent=self)
        self.__init_form__(self.__form)
        self.__form.value = SelectorValue.LOCAL

        # MUST BE INITIALIZED IN __init_actions__ !
        self.__progress_bar: QtWidgets.QProgressBar = typing.cast(QtWidgets.QProgressBar, None)
        self.__reject_button: QtWidgets.QPushButton = typing.cast(QtWidgets.QPushButton, None)
        self.__accept_button: QtWidgets.QPushButton = typing.cast(QtWidgets.QPushButton, None)

        init_header: typing.Optional[typing.Callable[[QtWidgets.QVBoxLayout], None]] = getattr(
            self,
            '__init_header__',
            None
        )
        init_footer: typing.Optional[typing.Callable[[QtWidgets.QVBoxLayout], None]] = getattr(
            self,
            '__init_footer__',
            None
        )

        dialog_layout: typing.Final[QtWidgets.QVBoxLayout] = QtWidgets.QVBoxLayout()
        dialog_layout.setContentsMargins(0, 0, 0, 0)

        if init_header is not None:
            init_header(dialog_layout)  # pylint: disable=not-callable
            dialog_layout.addWidget(widgets.SpacerVertical())
        dialog_layout.addLayout(self.__form.layout)
        dialog_layout.addWidget(widgets.SpacerVertical())
        if init_footer is not None:
            init_footer(dialog_layout)  # pylint: disable=not-callable
            dialog_layout.addWidget(widgets.SpacerVertical())
        self.__init_actions__(dialog_layout)

        # events, that require layout to be ready
        self.__form.sig_value_changed.connect(self._process_selection)
        self._process_selection(self.__form.value)

        self.setLayout(dialog_layout)
        self.setMinimumWidth(common.EnvironmentActionsDialog.BASE_DIALOG_WIDTH)

    def __init_form__(self, form: SelectorForm) -> None:
        """
        Initialize form part of the dialog.

        This method can also call other methods for additional controls for each option (e.g.
        :code:`__init_local_form__` or :code:`__init_cloud_form__`.
        """
        record: SelectorRecord
        for record in [
            SelectorRecord(value=SelectorValue.LOCAL, title='Local drive'),
            SelectorRecord(value=SelectorValue.CLOUD, title='Anaconda Cloud'),
        ]:
            option: SelectorOption = SelectorOption(name=record.title)
            form[record.value] = option
            self._controls.all += utilities.WidgetGroup(option.radio, option.label)

            initializer: typing.Optional[typing.Callable[[SelectorForm], None]] = getattr(
                self,
                f'__init_{record.value.name.lower()}_form__',
                None,
            )
            if initializer is not None:
                initializer(form)

    def __init_actions__(
            self,
            layout: QtWidgets.QVBoxLayout,
            *,
            accept_text: str = 'Accept',
            reject_text: str = 'Cancel',
    ) -> None:
        """Initialize actions part of the dialog."""
        # Progress bar
        progress_frame: typing.Final[ProgressFrame] = ProgressFrame()
        self.__progress_bar = progress_frame.progress_bar

        # Buttons
        self.__reject_button = widgets.ButtonNormal()
        self.__reject_button.setText(reject_text)
        self.__reject_button.clicked.connect(self._process_reject)

        self.__accept_button = widgets.ButtonPrimary()
        self.__accept_button.setDefault(True)
        self.__accept_button.setText(accept_text)
        self.__accept_button.clicked.connect(self._process_accept)

        # Layout
        container: typing.Final[QtWidgets.QHBoxLayout] = QtWidgets.QHBoxLayout()
        container.addWidget(progress_frame, 1, QtCore.Qt.AlignVCenter)
        container.addWidget(self.__reject_button, 0, QtCore.Qt.AlignVCenter)
        container.addWidget(self.__accept_button, 0, QtCore.Qt.AlignVCenter)
        container.setContentsMargins(0, 0, 0, 0)
        container.setSpacing(12)

        layout.addLayout(container)

        self._controls.all += utilities.WidgetGroup(self.__accept_button)

    @property
    def _controls(self) -> SelectorDialogControls:  # noqa: D401
        """
        Collection of controls in the dialog.

        All controls should be registered here, so they might be disabled/enabled on dialog changes (selection of
        another account option, or starting a process).
        """
        return self.__controls

    @property
    def outcome(self) -> SelectorOutcome:  # noqa: D401
        """Result of dialog execution."""
        return self.__outcome

    @property
    def selection(self) -> SelectorValue:  # noqa: D401
        """
        Current selected value.

        Might not correspond to the actual dialog :meth:`~SelectorDialog.value`.
        """
        return self.__form.value

    @property
    def value(self) -> SelectorValue:  # noqa: D401
        """Selected target in the dialog."""
        if self.__value is None:
            raise AttributeError('value is not available')
        return self.__value

    @value.setter
    def value(self, value: SelectorValue) -> None:
        """Update dialog `value`."""
        self.__form.value = value

    def set_acceptable(self, state: bool = True) -> None:
        """Set state for accept button (enabled or not)."""
        self.__accept_button.setEnabled(state)

    def set_busy(self, state: bool = True) -> None:
        """Set overall dialog state (any process is going on and controls should be disabled or not)."""
        if state:
            self._controls.suspend()
        else:
            self._controls.restore()

        if self.__progress_bar is not None:
            self.__progress_bar.setVisible(state)

    def set_rejectable(self, state: bool = True) -> None:
        """Set state for reject button (enabled or not)."""
        self.__reject_button.setEnabled(state)

    def _process_accept(self) -> None:
        """Process clicking on the 'OK' button."""
        self.__value = self.__form.value
        self.__outcome = SelectorOutcome.ACCEPT
        self.accept()

    def _process_link(self, link: str) -> None:
        """Process clicking on the label links."""
        if link == 'navigator://cloud/login':
            self.__value = SelectorValue.CLOUD
            self.__outcome = SelectorOutcome.LOGIN_REQUEST
            self.accept()
            return

        raise ValueError(f'unexpected link value to process: {link!r}')

    def _process_reject(self) -> None:
        """Process clicking on the 'Cancel' button."""
        self.__outcome = SelectorOutcome.REJECT
        self.reject()

    def _process_selection(self, value: SelectorValue) -> None:
        """Process changing selected value in the dialog."""
        self._controls.account_local.enable(value == SelectorValue.LOCAL)
        self._controls.account_cloud.enable(value == SelectorValue.CLOUD)


class MessageType(enum.IntEnum):
    """
    How to treat a message content for errors.

    .. py:attribute:: ABSOLUTE

        Message must be shown in any situation.

    .. py:attribute:: FALLBACK

        Show message only if there is no other message detected automatically.
    """

    ABSOLUTE = enum.auto()
    FALLBACK = enum.auto()


def retrieve_message(
        exception: BaseException,
        message_content: str,
        message_type: MessageType = MessageType.FALLBACK,
) -> str:
    """
    Prepare message for error.

    :param exception: Exception to retrieve error message from.
    :param message_content: Additional message to show in specific case.
    :param message_type: Case to show `message_content` in.
    :return: Detected message.
    """
    if message_type == MessageType.ABSOLUTE:
        return message_content

    if isinstance(exception, requests.RequestException) and (exception.response is not None):
        try:
            return html.escape(exception.response.json()['error']['message'])
        except (ValueError, TypeError, KeyError):
            pass

    if message_type == MessageType.FALLBACK:
        return message_content

    raise NotImplementedError()


class PrepopulatedSelectorDialog(SelectorDialog):
    """Customized :class:`~SelectorDialog` with additional controls for environment name."""

    def __init_header__(
            self,
            layout: QtWidgets.QVBoxLayout,
            *,
            caption_text: str = 'Options:',
    ) -> None:
        """Initialize header part of the dialog."""
        caption: typing.Final[QtWidgets.QLabel] = widgets.LabelBase()
        caption.setText(
            f'<span style="font-size: 16px; font-weight: 500">{html.escape(caption_text)}</span>',
        )

        self.__heading_errors: QtWidgets.QVBoxLayout  # pylint: disable=attribute-defined-outside-init
        self.__heading_errors = QtWidgets.QVBoxLayout()  # pylint: disable=attribute-defined-outside-init
        self.__heading_errors.setContentsMargins(0, 0, 0, 0)
        self.__heading_errors.setSpacing(0)

        layout.addWidget(caption)
        layout.addWidget(widgets.SpacerVertical())
        layout.addLayout(self.__heading_errors)

        self._controls.all += utilities.WidgetGroup(caption)

    def __init_cloud_form__(self, form: SelectorForm) -> None:
        """Initialize additional controls for Cloud option."""
        account: typing.Optional[str] = cloud.CloudAPI().username

        self.__cloud_account: QtWidgets.QLabel = widgets.LabelBase()  # pylint: disable=attribute-defined-outside-init
        self.__cloud_account.linkActivated.connect(self._process_link)
        if account:
            self.__cloud_account.setText(
                f'<span style="font-size: 12px; font-weight: 500">You are signed in as {html.escape(account)}</span>'
            )
        else:
            self.__cloud_account.setText(
                '<span style="font-size: 12px; font-weight: 500">'  # pylint: disable=implicit-str-concat
                '<a href="navigator://cloud/login" style="color: #43B049; text-decoration: none">Sign in</a>'
                ' to save your environment</span>'
            )

        form[SelectorValue.CLOUD].enabled = bool(account)
        form.add_widget(self.__cloud_account, 1, 1, 1, QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)

        self._controls.all += utilities.WidgetGroup(self.__cloud_account)

    def __init_footer__(
            self,
            layout: QtWidgets.QVBoxLayout,
            *,
            caption_text: str = 'Environment:',
    ) -> None:
        """Initialize footer part of the dialog."""
        caption: typing.Final[QtWidgets.QLabel] = widgets.LabelBase()
        caption.setText(
            f'<span style="font-size: 16px; font-weight: 500">{html.escape(caption_text)}</span>',
        )

        self.__environment_name: common.LineEditEnvironment  # pylint: disable=attribute-defined-outside-init
        self.__environment_name = common.LineEditEnvironment()  # pylint: disable=attribute-defined-outside-init
        self.__environment_name.textChanged.connect(self._process_environment_name)

        self.__environment_override: widgets.CheckBoxBase  # pylint: disable=attribute-defined-outside-init
        self.__environment_override = widgets.CheckBoxBase()  # pylint: disable=attribute-defined-outside-init
        self.__environment_override.setText('Overwrite existing environment')
        self.__environment_override.stateChanged.connect(self._process_environment_override)

        self.__footer_error: global_common.WarningLabel  # pylint: disable=attribute-defined-outside-init
        self.__footer_error = global_common.WarningLabel()  # pylint: disable=attribute-defined-outside-init

        content: typing.Final[QtWidgets.QVBoxLayout] = QtWidgets.QVBoxLayout()
        content.addWidget(self.__environment_name)
        content.addWidget(self.__environment_override)
        content.addWidget(self.__footer_error)
        content.setContentsMargins(20, 0, 0, 0)
        content.setSpacing(8)

        layout.addWidget(caption)
        layout.addWidget(widgets.SpacerVertical())
        layout.addLayout(content)

        self._controls.all += utilities.WidgetGroup(caption, self.__environment_name, self.__environment_override)

    @property
    def environment_name(self) -> str:  # noqa: D401
        """Chosen name of the new environment."""
        return self.__environment_name.text()

    @environment_name.setter
    def environment_name(self, value: str) -> None:
        """Update `environment_name` value."""
        # `self.__environment_name.setText()` skips value validation and allows setting invalid value
        # `self.__environment_name.insert(value)` discards whole value if it is not valid
        #
        # per-character insertions allows discarding only invalid characters

        character: str
        self.__environment_name.selectAll()
        for character in value:
            self.__environment_name.insert(character)

    @property
    def environment_override(self) -> bool:  # noqa: D401
        """Should existing environment  (if such) be overridden or not."""
        return self.__environment_override.isChecked()

    @environment_override.setter
    def environment_override(self, value: bool) -> None:
        """Update `environment_override` value."""
        self.__environment_override.setChecked(value)

    @property
    def footer_error(self) -> str:  # noqa: D401
        """Content of the error message in the footer."""
        return self.__footer_error.text

    @footer_error.setter
    def footer_error(self, value: str) -> None:
        """Update `footer_error` value."""
        self.__footer_error.text = value

    @property
    def cloud_account(self) -> str:  # noqa: D401
        """Text of the cloud account state label."""
        return self.__cloud_account.text()

    @cloud_account.setter
    def cloud_account(self, value: str) -> None:
        """Update `cloud_account` value."""
        self.__cloud_account.setText(value)

    def add_heading_error(self, value: str) -> None:
        """Add new error block to the header."""
        self.__heading_errors.addWidget(global_common.WarningBlock(text=value))

    def clear_heading_errors(self) -> None:
        """Remove all error blocks from the header."""
        item: typing.Optional[QtWidgets.QLayoutItem] = self.__heading_errors.takeAt(0)
        while item is not None:
            item.widget().deleteLater()
            item = self.__heading_errors.takeAt(0)

    def _process_environment_name(self, value: str) -> None:
        """Process change of the `environment_name` value."""

    def _process_environment_override(self, value: int) -> None:
        """Process change of the `environment_override` value."""

    # error handlers

    def _handle_header_error(
            self,
            message_content: str,
            message_type: MessageType = MessageType.FALLBACK,
    ) -> 'cloud_error_parsers.Handler[BaseException]':
        """Handle an error, and add error block to the header."""
        def result(exception: BaseException) -> bool:
            self.add_heading_error(HEADING_ERROR_TEMPLATE.format(
                content=retrieve_message(exception, message_content, message_type),
            ))
            return False

        return result

    def _handle_footer_error(
            self,
            message_content: str,
            message_type: MessageType = MessageType.FALLBACK,
    ) -> 'cloud_error_parsers.Handler[BaseException]':
        """Handle an error, and add error message to the footer."""
        def result(exception: BaseException) -> bool:
            self.footer_error = FOOTER_ERROR_TEMPLATE.format(
                content=retrieve_message(exception, message_content, message_type),
            )
            return False

        return result
