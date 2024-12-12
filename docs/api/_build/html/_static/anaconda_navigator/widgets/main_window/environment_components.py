# -*- coding: utf-8 -*-

# pylint: disable=protected-access

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Components for environment management."""

__all__ = ['EnvironmentsComponent']

import datetime
import os
import typing
import uuid
from qtpy import QtCore
from anaconda_navigator.utils import constants
from anaconda_navigator.utils import telemetry
from .. import dialogs
from ..dialogs import environment as environment_dialogs
from ..dialogs import packages as package_dialogs
from ..tabs import environments as environment_tabs
from . import common

if typing.TYPE_CHECKING:
    from qtpy import QtWidgets
    from anaconda_navigator.widgets import main_window
    from anaconda_navigator.widgets.dialogs.environment import import_dialogs


class SelectorInitializer(typing.Protocol):  # pylint: disable=too-few-public-methods
    """Common interface for SelectorDialogs."""

    def __call__(
        self,
        parent: typing.Optional['QtWidgets.QWidget'] = None,
        initial: environment_dialogs.SelectorValue = environment_dialogs.SelectorValue.LOCAL,
    ) -> environment_dialogs.SelectorDialog:
        """Initialize new SelectorDialog instance."""


class EnvironmentsComponent(common.Component):
    """Component for environment management."""

    __alias__ = 'environments'

    def __init__(self, parent: 'main_window.MainWindow') -> None:
        """Initialize new :class:`~EnvironmentsComponent` instance."""
        super().__init__(parent=parent)

        self.__environments: typing.Any = None

        self.__timer = QtCore.QTimer()  # Check for new environments
        self.__timer.setInterval(16000)
        self.__timer.timeout.connect(self.check_for_new_environments)

        self.__tab = environment_tabs.EnvironmentsTab(parent=self.main_window)
        self.__tab.sig_channels_requested.connect(self.main_window.show_channels)
        self.__tab.sig_update_index_requested.connect(self.main_window.update_index)
        self.__tab.sig_item_selected.connect(self.main_window.select_environment)
        self.__tab.sig_ready.connect(lambda: self.main_window.set_busy_status(conda=False))
        self.__tab.sig_cancel_requested.connect(self.main_window.show_cancel_process)
        self.__tab.sig_create_requested.connect(self.show_create_environment)
        self.__tab.sig_clone_requested.connect(self.show_clone_environment)
        self.__tab.sig_backup_requested.connect(self.show_backup_environment)
        self.__tab.sig_import_requested.connect(self.show_import_environment)
        self.__tab.sig_remove_requested.connect(self.show_remove_environment)
        self.__tab.sig_packages_action_requested.connect(self.show_conda_packages_action)

        self.main_window.stack.addTab(self.__tab, text='Environments')
        self.main_window.sig_logged_in.connect(lambda: self.__tab.sig_update_index_requested.emit('Login'))
        self.main_window.sig_logged_out.connect(lambda: self.__tab.sig_update_index_requested.emit('Logout'))

    @property
    def environments(self) -> typing.Any:
        """Collection of detected environments."""
        return self.__environments

    @property
    def tab(self) -> typing.Any:
        """Primary tab control."""
        return self.__tab

    def check_for_new_environments(self):
        """Check for new environments periodically on the system."""
        def listener(worker: typing.Any, output: typing.Any, error: str) -> None:  # pylint: disable=unused-argument
            result = output.get('__environments')
            if (result is not None) and (self.__environments != result):
                self.__environments = result
                self.main_window.select_environment(prefix=self.main_window.current_prefix)

        conda_worker = self.main_window.api.conda_info(prefix=self.main_window.current_prefix)
        conda_worker.sig_chain_finished.connect(listener)

    def show_create_environment(self) -> None:
        """Create new basic environment with selectable python version."""
        telemetry.ANALYTICS.instance.event('navigate', {'location': '/environments/create_environment'})

        dialog = environment_dialogs.CreateDialog(parent=self.main_window, api=self.main_window.api)
        worker_info = self.main_window.api.conda_data()
        worker_info.sig_chain_finished.connect(dialog.setup)
        if dialog.exec_():
            name = dialog.name
            prefix = dialog.prefix
            if name and prefix:
                self.main_window.set_busy_status(conda=True)
                worker = self.main_window.api.create_environment(prefix=prefix, packages=dialog.packages)
                worker.sig_partial.connect(self.main_window._conda_partial_output_ready)
                worker.sig_finished.connect(self.main_window._conda_output_ready)

                # Common tasks for tabs and widgets on tabs
                self.main_window.update_status(action=worker.action_msg, value=0, max_value=0)
                self.main_window.current_prefix = prefix
                self.main_window.set_widgets_enabled(False)
                self.__tab.add_temporal_item(name)

        telemetry.ANALYTICS.instance.event('navigate', {'location': '/environments'})

    def show_clone_environment(self) -> None:
        """Clone currently selected environment."""
        clone_from_prefix = self.main_window.current_prefix
        clone_from_name = os.path.basename(clone_from_prefix)
        telemetry.ANALYTICS.instance.event('navigate', {'location': '/environments/clone_environment'})
        dialog = environment_dialogs.CloneDialog(parent=self.main_window, clone_from_name=clone_from_name)
        worker_info = self.main_window.api.conda_data()
        worker_info.sig_chain_finished.connect(dialog.setup)

        if dialog.exec_():
            name = dialog.name
            prefix = dialog.prefix

            if name and prefix:
                self.main_window.set_busy_status(conda=True)
                worker = self.main_window.api.clone_environment(clone_from_prefix=clone_from_prefix, prefix=prefix)
                worker.sig_partial.connect(self.main_window._conda_partial_output_ready)
                worker.sig_finished.connect(self.main_window._conda_output_ready)

                # Actions on tabs and subwidgets
                self.main_window.update_status(action=worker.action_msg, value=0, max_value=0)
                self.main_window.set_widgets_enabled(False)
                self.__tab.add_temporal_item(name)

        telemetry.ANALYTICS.instance.event('navigate', {'location': '/environments'})

    def show_backup_environment(self) -> None:
        """Create backup of current active environment."""
        telemetry.ANALYTICS.instance.event('navigate', {'location': '/environments/backup_environment'})

        value: environment_dialogs.SelectorValue = environment_dialogs.SelectorValue.CLOUD
        environment_name: str = (
            f'{os.path.basename(self.main_window.current_prefix)}'
            '_'
            f'{datetime.date.today().strftime("%Y%m%d")}'
        )
        while True:
            dialog: environment_dialogs.BackupSelectorDialog = environment_dialogs.BackupSelectorDialog(
                parent=self.main_window,
            )
            dialog.environment_name = environment_name
            try:
                dialog.value = value
            except ValueError:
                pass
            dialog.exec_()

            if dialog.outcome == environment_dialogs.SelectorOutcome.REJECT:
                break

            value = dialog.value
            environment_name = dialog.environment_name

            if dialog.outcome == environment_dialogs.SelectorOutcome.LOGIN_REQUEST:
                if value == environment_dialogs.SelectorValue.CLOUD:
                    self.main_window.components.accounts.log_into_cloud()
                    continue

                raise NotImplementedError(f'Can not login to {value.name}')

            if dialog.outcome == environment_dialogs.SelectorOutcome.ACCEPT:
                dialogs.MessageBox(
                    dialogs.MessageBox.INFORMATION_BOX,
                    title='Backup Environment',
                    text='Environment backed up successfully',
                    parent=self.main_window,
                ).exec_()
                break

            raise NotImplementedError()

        telemetry.ANALYTICS.instance.event('navigate', {'location': '/environments'})

    def show_import_environment(self) -> None:  # pylint: disable=missing-function-docstring,too-many-statements
        telemetry.ANALYTICS.instance.event('navigate', {'location': '/environments/import_environment'})

        value: environment_dialogs.SelectorValue = environment_dialogs.SelectorValue.CLOUD
        while True:
            dialog: environment_dialogs.ImportSelectorDialog = environment_dialogs.ImportSelectorDialog(
                parent=self.main_window,
            )
            try:
                dialog.value = value
            except ValueError:
                pass
            dialog.exec_()

            if dialog.outcome == environment_dialogs.SelectorOutcome.REJECT:
                break

            value = dialog.value

            if dialog.outcome == environment_dialogs.SelectorOutcome.LOGIN_REQUEST:
                if value == environment_dialogs.SelectorValue.CLOUD:
                    self.main_window.components.accounts.log_into_cloud()
                    continue

                raise NotImplementedError(f'Can not login to {value.name}')

            if dialog.outcome == environment_dialogs.SelectorOutcome.ACCEPT:
                conda_details: 'import_dialogs.CondaDetails' = typing.cast(
                    'import_dialogs.CondaDetails',
                    dialog.conda_details,
                )

                old_prefix: typing.Optional[str] = conda_details.environments.get(dialog.environment_name, None)
                new_prefix: str = (
                    old_prefix
                    or os.path.join(conda_details.environment_directories[0], dialog.environment_name)
                )
                file: str = typing.cast(str, dialog.environment_specification)
                file_temporary: bool = bool(dialog.environment_temporary)

                def env_validate() -> None:
                    self.main_window.set_busy_status(conda=True)
                    worker = self.main_window.api.import_environment(
                        prefix=os.path.join(conda_details.environment_directories[0], uuid.uuid4().hex),
                        file=file,
                        validate_only=True,
                    )
                    worker.sig_partial.connect(self.main_window._conda_partial_output_ready)
                    worker.sig_finished.connect(env_remove)

                    self.main_window.update_status(action=worker.action_msg, value=0, max_value=0)
                    self.main_window.set_widgets_enabled(False)

                def env_remove(worker: typing.Any, output: typing.Any, error: typing.Any) -> None:
                    if error:
                        worker.old_prefix = self.main_window.current_prefix  # to prevent switch to default prefix
                        self.main_window._conda_output_ready(worker, output, error)
                        return

                    worker = self.main_window.api.remove_environment(prefix=old_prefix)
                    worker.sig_partial.connect(self.main_window._conda_partial_output_ready)
                    worker.sig_finished.connect(env_import)

                    self.main_window.update_status(action=worker.action_msg, value=0, max_value=0)

                def env_import(*args: typing.Any) -> None:  # pylint: disable=unused-argument
                    self.main_window.set_busy_status(conda=True)
                    worker = self.main_window.api.import_environment(prefix=new_prefix, file=file)
                    worker.sig_partial.connect(self.main_window._conda_partial_output_ready)
                    worker.sig_finished.connect(self.main_window._conda_output_ready)
                    worker.sig_finished.connect(cleanup)

                    self.main_window.update_status(action=worker.action_msg, value=0, max_value=0)
                    if old_prefix is None:
                        self.__tab.add_temporal_item(os.path.basename(new_prefix))
                    self.main_window.set_widgets_enabled(False)

                def cleanup(*args: typing.Any) -> None:  # pylint: disable=unused-argument
                    if file_temporary:
                        try:
                            os.remove(file)
                        except OSError:
                            pass

                if old_prefix is not None:
                    env_validate()
                else:
                    env_import()
                break

            raise NotImplementedError()

        telemetry.ANALYTICS.instance.event('navigate', {'location': '/environments'})

    def show_remove_environment(self) -> None:
        """Clone currently selected environment."""
        prefix = self.main_window.current_prefix
        name = os.path.basename(prefix)

        if prefix != self.main_window.api.ROOT_PREFIX:
            telemetry.ANALYTICS.instance.event('navigate', {'location': '/environments/remove_environment'})
            dialog = environment_dialogs.RemoveDialog(parent=self.main_window, name=name, prefix=prefix)
            if dialog.exec_():
                self.main_window.set_busy_status(conda=True)

                if prefix == self.main_window.config.get('main', 'default_env'):
                    self.main_window.config.set('main', 'default_env', self.main_window.api.ROOT_PREFIX)

                worker = self.main_window.api.remove_environment(prefix=prefix)
                worker.sig_partial.connect(self.main_window._conda_partial_output_ready)
                worker.sig_finished.connect(self.main_window._conda_output_ready)

                # Actions on tabs and subwidgets
                self.main_window.update_status(action=worker.action_msg, value=0, max_value=0)
                self.main_window.set_widgets_enabled(False)
            telemetry.ANALYTICS.instance.event('navigate', {'location': '/environments'})

    def show_conda_packages_action(  # pylint: disable=too-many-branches,too-many-locals,too-many-statements
            self,
            conda_packages_actions: typing.Mapping[int, typing.Sequence[typing.Mapping[str, typing.Any]]],
            pip_packages_actions: typing.Mapping[int, typing.Sequence[typing.Mapping[str, typing.Any]]],
    ) -> None:
        """Process the coda actions on the packages for current environment."""
        install_packages = []
        remove_packages = []
        update_packages = []
        remove_pip_packages = []
        for action in [constants.ACTION_DOWNGRADE, constants.ACTION_INSTALL, constants.ACTION_UPGRADE]:
            pkgs_action = conda_packages_actions[action]
            for pkg in pkgs_action:
                name = pkg['name']
                version = pkg['version_to']
                if version:
                    spec = name + '==' + version
                else:
                    spec = name
                install_packages.append(spec)

        for pkg in conda_packages_actions[constants.ACTION_REMOVE]:
            remove_packages.append(pkg['name'])

        for pkg in conda_packages_actions[constants.ACTION_UPDATE]:
            update_packages.append(pkg['name'])

        for pkg in pip_packages_actions[constants.ACTION_REMOVE]:
            remove_pip_packages.append(pkg['name'])

        telemetry.ANALYTICS.instance.event('navigate', {'location': '/environments/package_actions'})
        self.main_window.set_busy_status(conda=True)
        if install_packages:
            pkgs = install_packages
            dialog = package_dialogs.PackagesDialog(
                parent=self.main_window,
                packages=pkgs,
            )
            worker_deps = self.main_window.api.install_packages(
                prefix=self.main_window.current_prefix,
                pkgs=pkgs,
                dry_run=True,
            )
        elif update_packages:
            pkgs = update_packages
            dialog = package_dialogs.PackagesDialog(
                parent=self.main_window,
                packages=pkgs,
                update_only=True,
            )
            worker_deps = self.main_window.api.update_packages(
                prefix=self.main_window.current_prefix,
                pkgs=pkgs,
                dry_run=True,
            )
        elif remove_packages:
            pkgs = remove_packages
            dialog = package_dialogs.PackagesDialog(
                parent=self.main_window,
                packages=pkgs,
                remove_only=True,
                pip_packages=remove_pip_packages,
            )
            worker_deps = self.main_window.api.remove_packages(
                prefix=self.main_window.current_prefix,
                pkgs=pkgs,
                dry_run=True,
            )

        worker_deps.prefix = self.main_window.current_prefix
        worker_deps.sig_finished.connect(dialog.setup)

        if dialog.exec_():
            worker = None
            if remove_packages:
                worker = self.main_window.api.remove_packages(prefix=self.main_window.current_prefix, pkgs=pkgs)
            elif install_packages:
                worker = self.main_window.api.install_packages(prefix=self.main_window.current_prefix, pkgs=pkgs)
            elif update_packages:
                worker = self.main_window.api.update_packages(prefix=self.main_window.current_prefix, pkgs=pkgs)
            elif remove_pip_packages:
                pass
                # Run pip command :-p?

            if worker:
                worker.sig_partial.connect(self.main_window._conda_partial_output_ready)
                worker.sig_finished.connect(self.main_window._conda_output_ready)
                self.main_window.set_widgets_enabled(False)
                self.main_window.set_busy_status(conda=True)
                self.main_window.update_status(action=worker.action_msg, value=0, max_value=0)
        else:
            if not worker_deps.is_finished():
                self.main_window.api.conda_terminate()
            self.main_window.set_busy_status(conda=False)

        telemetry.ANALYTICS.instance.event('navigate', {'location': '/environments'})

    # Virtual endpoints

    def setup(self, worker: typing.Any, output: typing.Any, error: str, initial: bool) -> None:
        """Perform component configuration from `conda_data`."""
        if initial:
            self.__environments = output.get('processed_info', {}).get('__environments')
        else:
            self.__tab.setup(output)

    def update_style_sheet(self) -> None:
        """Update style sheet of the tab."""
        self.__tab.update_style_sheet()

    def start_timers(self) -> None:
        """Start component timers."""
        self.__timer.start()

    def stop_timers(self) -> None:
        """Stop component timers."""
        self.__timer.stop()
