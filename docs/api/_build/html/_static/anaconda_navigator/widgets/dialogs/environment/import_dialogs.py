# -*- coding: utf-8 -*-

# pylint: disable=attribute-defined-outside-init,unused-argument

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Dialogs for importing environments."""

__all__ = ['ImportSelectorDialog']

import enum
import html
import os
import tempfile
import typing
import uuid

from qtpy import compat
from qtpy import QtCore
from qtpy import QtWidgets
import requests
import yaml

from anaconda_navigator.api import cloud
from anaconda_navigator.api.cloud import tools as cloud_tools
from anaconda_navigator.utils import styles
from anaconda_navigator.utils import telemetry
from anaconda_navigator.utils import workers
from anaconda_navigator import widgets
from anaconda_navigator.widgets import dialogs
from anaconda_navigator.widgets import common as global_common
from anaconda_navigator.widgets.lists import environments
from .. import utilities
from . import common
from . import multiaccount_dialogs

if typing.TYPE_CHECKING:
    from anaconda_navigator.widgets import main_window


DEFAULT_ERROR: typing.Final[str] = 'Unable to import environment due to an unknown error.'

ENVIRONMENT_FETCH_LIMIT: typing.Final[int] = 20
ENVIRONMENT_ORDER: typing.Final[cloud.EnvironmentSort] = cloud.EnvironmentSort.UPDATED_DESC


def env_to_spec(source: str, target: str, name: typing.Optional[str] = None) -> None:
    """
    Convert Conda environment file to Conda specification file.

    :param source: Path to the conda environment file.
    :param target: Path to the conda specification file.
    :param name: Name of the environment being converted.
    """
    content: typing.Any

    stream: typing.TextIO
    with open(source, 'rt', encoding='utf-8') as stream:
        content = yaml.safe_load(stream)

    content['name'] = name

    with open(target, 'wt', encoding='utf-8') as stream:
        yaml.dump(content, stream, default_flow_style=False)


def pip_to_spec(source: str, target: str, name: typing.Optional[str] = None) -> None:
    """
    Convert pip requirements file to Conda specification file.

    :param source: Path to the pip requirements file.
    :param target: Path to the conda specification file.
    :param name: Name of the environment being converted.
    """
    dependencies: typing.List[str]

    stream: typing.TextIO
    with open(source, 'rt', encoding='utf-8') as stream:
        dependencies = [
            line
            for line in map(str.strip, stream.read().splitlines())
            if line[:1] not in ('', '#')
        ]

    with open(target, 'wt', encoding='utf-8') as stream:
        yaml.dump(
            {
                'name': name,
                'dependencies': [
                    'python',
                    {'pip': dependencies},
                ],
            },
            stream,
            default_flow_style=False,
        )


class ImportSelectorType(enum.IntEnum):
    """
    Type of the file, imported from the file system.

    .. py:attribute:: NONE

        File is not yet selected.

    .. py:attribute:: ENVIRONMENT

        Conda environment file.

    .. py:attribute:: SPECIFICATION

        Conda specification file.

    .. py:attribute:: PIP

        Pip requirements file.
    """

    NONE = enum.auto()
    ENVIRONMENT = enum.auto()
    SPECIFICATION = enum.auto()
    PIP = enum.auto()


class AccountForm(QtWidgets.QHBoxLayout):
    """Group of controls for opening external items."""

    def __init__(self) -> None:
        """Initialize new :class:`~AccountForm` instance."""
        super().__init__()

        self.__edit: typing.Final[widgets.LineEditBase] = widgets.LineEditBase()
        self.__edit.setReadOnly(True)

        self.__button: typing.Final[common.OpenIconButton] = common.OpenIconButton()

        self.addWidget(self.__edit)
        self.addWidget(self.__button)
        self.setSpacing(8)
        self.setContentsMargins(0, 8, 0, 0)

    @property
    def button(self) -> common.OpenIconButton:  # noqa: D401
        """Button for "open" action."""
        return self.__button

    @property
    def edit(self) -> widgets.LineEditBase:  # noqa: D401
        """Text control for selected value."""
        return self.__edit

    @property
    def group(self) -> utilities.WidgetGroup:  # noqa: D401
        """Group with dialog controls."""
        return utilities.WidgetGroup(self.edit, self.button)


class CondaDetails:
    """
    Environment-related details, retrieved from Conda.

    It should be parsed from `conda_data` results.
    """

    __slots__ = ('__environment_directories', '__environments')

    def __init__(self, source: typing.Mapping[str, typing.Any]) -> None:
        """Initialize new :class:`~CondaDetails` instance."""
        if not source:
            raise TypeError()
        try:
            self.__environment_directories: typing.Final[typing.Sequence[str]] = (
                source['processed_info']['__envs_dirs_writable']
            )
            self.__environments: typing.Final[typing.Mapping[str, str]] = {
                key: value
                for value, key in source['processed_info']['__environments'].items()
            }
        except (TypeError, ValueError, LookupError):
            raise ValueError() from None
        if not self.__environment_directories:
            raise ValueError()

    @property
    def environment_directories(self) -> typing.Sequence[str]:  # noqa: D401
        """List of directories, where new environments might be placed."""
        return self.__environment_directories

    @property
    def environments(self) -> typing.Mapping[str, str]:  # noqa: D401
        """Mapping of environment names to their prefixes."""
        return self.__environments


class ImportSelectorDialog(  # pylint: disable=too-many-instance-attributes
    multiaccount_dialogs.PrepopulatedSelectorDialog
):
    """Dialog for selecting the target to which to export an environment."""

    def __init__(self, parent: 'main_window.MainWindow') -> None:
        """Initialize new :class:`~BackupSelectorDialog` instance."""
        self.__conda_details: typing.Optional[CondaDetails] = None
        conda_worker = parent.api.conda_data()
        conda_worker.sig_chain_finished.connect(self.__parse_conda)

        self.__cloud_content: typing.Optional[typing.Mapping[str, typing.Any]] = None
        self.__cloud_controls: utilities.WidgetGroup = utilities.WidgetGroup()

        self.__environment_specification: typing.Optional[str] = None
        self.__environment_temporary: bool = False

        super().__init__(parent=parent)

        self.setWindowTitle('Import Environment')

    def __init_header__(  # pylint: disable=useless-super-delegation
            self,
            layout: QtWidgets.QVBoxLayout,
            *,
            caption_text: str = 'Import from:',
    ) -> None:
        """Initialize header part of the dialog."""
        super().__init_header__(layout, caption_text=caption_text)

    def __init_local_form__(self, form: multiaccount_dialogs.SelectorForm) -> None:
        """Initialize additional controls for local option."""
        content: typing.Final[AccountForm] = AccountForm()

        self.__local_environment: widgets.LineEditBase = content.edit
        self.__local_environment_type: ImportSelectorType = ImportSelectorType.NONE
        content.edit.textChanged.connect(self._process_local_environment)
        content.button.clicked.connect(self._process_open_local)

        form.add_layout(content, 1)
        self._controls.all += content.group

    def __init_cloud_form__(self, form: multiaccount_dialogs.SelectorForm) -> None:
        """Initialize additional controls for Cloud option."""
        super().__init_cloud_form__(form)

        content: typing.Final[AccountForm] = AccountForm()

        self.__cloud_environment: widgets.LineEditBase = content.edit
        content.edit.textChanged.connect(self._process_cloud_environment)
        content.button.clicked.connect(self._process_open_cloud)

        form.add_layout(content, 1)
        self._controls.all += content.group
        self.__cloud_controls = content.group

        self.__cloud_controls.disable()
        if cloud.CloudAPI().username:
            worker: workers.TaskWorker = cloud.CloudAPI().list_environments.worker(  # pylint: disable=no-member
                limit=ENVIRONMENT_FETCH_LIMIT,
                offset=0,
                sort=ENVIRONMENT_ORDER,
            )
            self.finished.connect(worker.cancel)
            worker.signals.sig_done.connect(lambda result: self.finished.disconnect(worker.cancel))
            worker.signals.sig_done.connect(self.__check_cloud_fetch)
            worker.start()

    def __init_footer__(  # pylint: disable=useless-super-delegation
            self,
            layout: QtWidgets.QVBoxLayout,
            *,
            caption_text: str = 'New environment name:',
    ) -> None:
        """Initialize footer part of the dialog."""
        super().__init_footer__(layout, caption_text=caption_text)

    def __init_actions__(  # pylint: disable=useless-super-delegation
            self,
            layout: QtWidgets.QVBoxLayout,
            *,
            accept_text: str = 'Import',
            reject_text: str = 'Cancel',
    ) -> None:
        """Initialize actions part of the dialog."""
        super().__init_actions__(layout, accept_text=accept_text, reject_text=reject_text)

    @property
    def conda_details(self) -> typing.Optional[CondaDetails]:  # noqa: D401
        """Environment details retrieved from Conda."""
        return self.__conda_details

    @property
    def environment_specification(self) -> typing.Optional[str]:  # noqa: D401
        """Path to environment specification file, which should be used to create new environment."""
        return self.__environment_specification

    @property
    def environment_temporary(self) -> bool:  # noqa: D401
        """Is `environment_specification` is a temporary file and should be removed after creating new environment."""
        return self.__environment_temporary

    @property
    def local_environment(self) -> str:  # noqa: D401
        """Selected local environment value."""
        return self.__local_environment.text()

    @local_environment.setter
    def local_environment(self, value: str) -> None:
        """Update `local_environment` value."""
        self.__local_environment.setText(value)

    @property
    def local_environment_type(self) -> ImportSelectorType:  # noqa: D401
        """Type of the selected `local_environment`."""
        return self.__local_environment_type

    @local_environment_type.setter
    def local_environment_type(self, value: ImportSelectorType) -> None:
        """Update `local_environment_type` value."""
        self.__local_environment_type = value

    @property
    def cloud_environment(self) -> str:  # noqa: D401
        """Selected Cloud environment value."""
        return self.__cloud_environment.text()

    @cloud_environment.setter
    def cloud_environment(self, value: str) -> None:
        """Update `cloud_environment` value."""
        self.__cloud_environment.setText(value)

    def __parse_conda(self, worker: typing.Any, output: typing.Any, error: typing.Any) -> None:
        """Parse response of the conda data command."""
        try:
            self.__conda_details = CondaDetails(source=output)
        except (TypeError, ValueError):
            pass
        else:
            self.__update_acceptable()

    def _process_accept(self) -> None:
        """Process clicking on the 'OK' button."""
        if self.conda_details is None:
            raise ValueError()
        if (not self.environment_override) and (self.environment_name in self.conda_details.environments):
            self.footer_error = multiaccount_dialogs.FOOTER_ERROR_TEMPLATE.format(
                content='Please rename environment to continue.',
            )
            return

        self.set_busy(True)
        self.clear_heading_errors()
        self.footer_error = ''

        if self.selection == multiaccount_dialogs.SelectorValue.LOCAL:
            self.__process_accept_local()
        elif self.selection == multiaccount_dialogs.SelectorValue.CLOUD:
            self.__process_accept_cloud()
        else:
            raise NotImplementedError()

    def _process_selection(self, value: multiaccount_dialogs.SelectorValue) -> None:
        """Process changing selected value in the dialog."""
        super()._process_selection(value)
        self.__update_acceptable()

    def _process_environment_name(self, value: str) -> None:
        """Process change of the `environment_name` value."""
        super()._process_environment_name(value)
        self.__update_acceptable()

    def _process_local_environment(self, value: str) -> None:
        """Process change of the `local_environment` value."""
        self.__update_acceptable()

    def _process_cloud_environment(self, value: str) -> None:
        """Process change of the `cloud_environment` value."""
        self.__update_acceptable()

    def __update_acceptable(self) -> None:
        """Update state of the accept button according to current dialog state."""
        if (self.__conda_details is not None) and self.environment_name:
            if (self.selection == multiaccount_dialogs.SelectorValue.LOCAL) and self.local_environment:
                self.set_acceptable(True)
                return

            if (self.selection == multiaccount_dialogs.SelectorValue.CLOUD) and self.cloud_environment:
                self.set_acceptable(True)
                return

        self.set_acceptable(False)

    # Local

    def _process_open_local(self) -> None:
        """Open dialog to select local environment for import."""
        environment_files: typing.Final[str] = 'Conda environment files (*.yaml *.yml)'
        specification_files: typing.Final[str] = 'Conda explicit specification files (*.txt)'
        pip_files: typing.Final[str] = 'Pip requirement files (*.txt)'

        path: str
        selected_filter: str
        path, selected_filter = compat.getopenfilename(
            parent=self,
            caption='Import Environment',
            basedir=os.path.expanduser('~'),
            filters=';;'.join([environment_files, specification_files, pip_files]),
        )
        if not path:
            return

        self.value = multiaccount_dialogs.SelectorValue.LOCAL

        self.__local_environment.setText(path)

        if selected_filter == environment_files:
            self.__local_environment_type = ImportSelectorType.ENVIRONMENT
        elif selected_filter == specification_files:
            self.__local_environment_type = ImportSelectorType.SPECIFICATION
        elif selected_filter == pip_files:
            self.__local_environment_type = ImportSelectorType.PIP

        self.environment_name = os.path.splitext(os.path.basename(path))[0]

    def __process_accept_local(self) -> None:
        """Process clicking on the 'OK' button when local account is selected."""
        telemetry.ANALYTICS.instance.event('request-environment-import', {'from': 'local'})
        if self.__local_environment_type == ImportSelectorType.SPECIFICATION:
            self.__environment_specification = self.local_environment
            self.__environment_temporary = False
            telemetry.ANALYTICS.instance.event('import-environment', {'from': 'local'})
            super()._process_accept()
            return

        path: str = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.yml')

        if self.__local_environment_type == ImportSelectorType.ENVIRONMENT:
            env_to_spec(source=self.local_environment, target=path, name=self.environment_name)

        if self.__local_environment_type == ImportSelectorType.PIP:
            pip_to_spec(source=self.local_environment, target=path, name=self.environment_name)

        self.__environment_specification = path
        self.__environment_temporary = True
        telemetry.ANALYTICS.instance.event('import-environment', {'from': 'local'})
        super()._process_accept()
        return

    # Cloud

    def __check_cloud_fetch(self, result: workers.TaskResult) -> None:
        """Parse available Cloud environments."""
        if result.status != workers.TaskStatus.SUCCEEDED:
            return

        try:
            self.__cloud_content = typing.cast(typing.Mapping[str, typing.Any], result.result)
        except workers.TaskCanceledError:
            self.__cloud_content = {'total': 0}

        total: int = self.__cloud_content['total']
        username: str = cloud.CloudAPI().username

        if total <= 0:
            self.cloud_account = (
                f'<span style="font-size: 12px; font-weight: 500">'
                f'No environments available for {html.escape(username)}'
                f'</span>'
            )
            return

        if total == 1:
            self.cloud_account = (
                f'<span style="font-size: 12px; font-weight: 500">'
                f'1 environment available for {html.escape(username)}'
                f'</span>'
            )
        else:
            self.cloud_account = (
                f'<span style="font-size: 12px; font-weight: 500">'
                f'{total} environments available for {html.escape(username)}'
                f'</span>'
            )
        self.__cloud_controls.enable()

    def _process_open_cloud(self) -> None:
        """Open dialog to select Cloud environment for import."""
        dialog: EnvSelectorDialog = EnvSelectorDialog(parent=self, envs=self.__cloud_content)
        if dialog.exec_():
            self.value = multiaccount_dialogs.SelectorValue.CLOUD
            self.environment_name = self.cloud_environment = dialog.current_item.name

    def __process_accept_cloud(self) -> None:
        """Process clicking on the 'OK' button when Cloud account is selected."""
        telemetry.ANALYTICS.instance.event('request-environment-import', {'from': 'cloud'})

        path: str = os.path.join(tempfile.gettempdir(), uuid.uuid4().hex + '.yml')

        worker: workers.TaskWorker = cloud.CloudAPI().download_environment.worker(  # pylint: disable=no-member
            name=self.__cloud_environment.text(),
            path=path,
        )
        self.finished.connect(worker.cancel)
        worker.signals.sig_done.connect(lambda result: self.finished.disconnect(worker.cancel))
        worker.signals.sig_done.connect(self.__check_cloud_download)
        worker.start()

    def __check_cloud_download(self, result: workers.TaskResult) -> None:
        """Check results of a Cloud environment download."""
        if not self.isVisible():
            return

        if result.status == workers.TaskStatus.SUCCEEDED:
            self.__environment_specification = result.call.kwargs['path']
            self.__environment_temporary = True
            telemetry.ANALYTICS.instance.event('import-environment', {'from': 'cloud'})
            super()._process_accept()

        elif result.status == workers.TaskStatus.FAILED:
            handlers: cloud_tools.HttpErrorHandlers = cloud_tools.HttpErrorHandlers()
            handlers.register_handler(BaseException, self._handle_header_error(DEFAULT_ERROR))
            handlers.handle(exception=typing.cast(BaseException, result.exception))

        self.set_busy(False)


class EnvSelectorItem(environments.BaseListItemEnv):  # pylint: disable=missing-class-docstring
    ENV_ITEM_HEIGHT = styles.SASS_VARIABLES.WIDGET_IMPORT_ENVIRONMENT_TOTAL_HEIGHT


class EnvSelectorDialog(dialogs.DialogBase):
    """List environments dialog."""

    sig_env_list_ready = QtCore.Signal(object)

    def __init__(
            self,
            parent: typing.Optional[QtWidgets.QWidget] = None,
            envs: typing.Optional[typing.Mapping[str, typing.Any]] = None,
    ) -> None:
        """List environments dialog."""
        super().__init__(parent=parent)

        self.api: typing.Final['cloud._CloudAPI'] = cloud.CloudAPI()
        self.offset: int = 0

        # Widgets
        heading_label: typing.Final[QtWidgets.QLabel] = widgets.LabelBase()
        heading_label.setText('<span style="font-size: 16px; font-weight: 500">Select environment from Cloud:</span>')

        self.list: typing.Final[environments.BaseListWidgetEnv] = environments.BaseListWidgetEnv()
        self.list.setFocus()

        self.error_label: typing.Final[global_common.WarningLabel] = global_common.WarningLabel()

        progress_frame: typing.Final[multiaccount_dialogs.ProgressFrame] = (
            multiaccount_dialogs.ProgressFrame()
        )
        self.progress_bar: typing.Final[QtWidgets.QProgressBar] = progress_frame.progress_bar

        self.button_cancel: typing.Final[widgets.ButtonNormal] = widgets.ButtonNormal('Cancel')

        self.button_ok: typing.Final[widgets.ButtonPrimary] = widgets.ButtonPrimary('Select')
        self.button_ok.setDisabled(True)

        # Layouts
        layout_buttons: typing.Final[QtWidgets.QHBoxLayout] = QtWidgets.QHBoxLayout()
        layout_buttons.addWidget(progress_frame, 1)
        layout_buttons.addWidget(self.button_cancel)
        layout_buttons.addWidget(self.button_ok)
        layout_buttons.setContentsMargins(0, 0, 0, 0)
        layout_buttons.setSpacing(12)

        layout: typing.Final[QtWidgets.QVBoxLayout] = QtWidgets.QVBoxLayout()
        layout.addWidget(heading_label)
        layout.addWidget(widgets.SpacerVertical())
        layout.addWidget(self.list)
        layout.addWidget(widgets.SpacerVertical())
        layout.addWidget(self.error_label)
        layout.addWidget(widgets.SpacerVertical())
        layout.addLayout(layout_buttons)

        # Setup
        self.setLayout(layout)
        self.setMinimumHeight(475)
        self.setMinimumWidth(460)
        self.setWindowTitle('Import New Environment')
        self.setFocus()

        # Signals
        self.sig_env_list_ready.connect(self.extend_list)
        self.button_ok.clicked.connect(self.accept)
        self.button_cancel.clicked.connect(self.reject)
        self.list.verticalScrollBar().valueChanged.connect(self.__extend_list_by_scroll)

        if envs:
            self.extend_list(envs)
        else:
            self._load_environments()

    @property
    def current_item(self) -> EnvSelectorItem:  # noqa: D401
        """Current selected item."""
        return self.list.currentItem()

    @staticmethod
    def get_env_names(envs: typing.Mapping[str, typing.Any]) -> typing.List[str]:
        """Get list of environment names."""
        try:
            return [
                env['name']
                for env in envs['items']
            ]
        except (TypeError, KeyError):
            return []

    def __extend_list_by_scroll(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """Download batch of environments and add their names to the list if scroll bar reached the bottom."""
        scroll_bar: QtWidgets.QScrollBar = self.list.verticalScrollBar()
        if scroll_bar.value() >= scroll_bar.maximum() - 3:
            self._load_environments()

    def __process_load_result(self, output: workers.TaskResult) -> None:
        """Check the output from worker and emit signal to extend list of environment names."""
        # if not self.isVisible():
        #     return

        self.progress_bar.setVisible(False)

        try:
            self.sig_env_list_ready.emit(output.result)
        except requests.RequestException:
            self.error_label.text = 'Unable to fetch available environments.'
        except workers.TaskCanceledError:
            pass
        else:
            if output.result.get('items'):
                self.list.verticalScrollBar().valueChanged.connect(self.__extend_list_by_scroll)

    def _load_environments(self) -> None:
        """Start loading a batch of environments in separate tread"""
        self.progress_bar.setVisible(True)
        self.list.verticalScrollBar().valueChanged.disconnect(self.__extend_list_by_scroll)

        worker: workers.TaskWorker = self.api.list_environments.worker(  # pylint: disable=no-member
            limit=ENVIRONMENT_FETCH_LIMIT,
            offset=self.offset,
            sort=ENVIRONMENT_ORDER,
        )
        worker.signals.sig_done.connect(self.__process_load_result)
        worker.start()

    def extend_list(self, envs: typing.Mapping[str, typing.Any]) -> None:
        """Add environment names to the list."""
        env_names: typing.Sequence[str] = self.get_env_names(envs)
        for name in env_names:
            self.list.addItem(EnvSelectorItem(name))

        self.offset += len(env_names)
        self.button_ok.setEnabled(self.offset > 0)
