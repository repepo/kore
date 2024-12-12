# -*- coding: utf-8 -*-

# pylint: disable=protected-access

# -----------------------------------------------------------------------------
# Copyright (c) 2016-2017 Anaconda, Inc.
#
# May be copied and distributed freely only as part of an Anaconda or
# Miniconda installation.
# -----------------------------------------------------------------------------

"""Components for environment management."""

__all__ = ['ApplicationsComponent']

import html
import os
import typing
import webbrowser
from qtpy import QtCore
from anaconda_navigator.api import external_apps
from anaconda_navigator import config as anaconda_config
from anaconda_navigator.utils import constants
from anaconda_navigator.utils import launch
from anaconda_navigator.utils import telemetry
from anaconda_navigator.utils import version_utils
from anaconda_navigator.widgets import dialogs
from anaconda_navigator.widgets.dialogs import environment as environment_dialogs
from . import common

if typing.TYPE_CHECKING:
    from anaconda_navigator.widgets import main_window


class ApplicationsComponent(common.Component):
    """Component for launching third-party applications."""

    __alias__ = 'applications'

    def __init__(self, parent: 'main_window.MainWindow') -> None:
        """Initialize new :class:`~EnvironmentsComponent` instance."""
        super().__init__(parent=parent)

        self.__running_processes: typing.Final[typing.List[launch.RunningProcess]] = []

        self.__listener_timer: typing.Final[QtCore.QTimer] = QtCore.QTimer()
        self.__listener_timer.timeout.connect(self.update_running_processes)
        self.__listener_timer.setInterval(5000)
        self.__listener_timer.start()

        self.__feedback_timer: typing.Final[QtCore.QTimer] = QtCore.QTimer()
        self.__feedback_timer.timeout.connect(self.__feedback_timeout)
        self.__feedback_timer.setSingleShot(True)
        self.__feedback_timer.setInterval(5000)

    @property
    def running_processes(self) -> typing.List[launch.RunningProcess]:
        """Collection of currently launched processes."""
        return self.__running_processes[:]

    def update_running_processes(self) -> None:
        """Update status of applications launched from Navigator."""
        index: int
        for index in reversed(range(len(self.__running_processes))):
            running_application: launch.RunningProcess = self.__running_processes[index]
            if running_application.return_code is None:
                continue

            if (running_application.return_code != 0) and (running_application.age.total_seconds() < 15.0):
                self.show_application_launch_errors(running_application)

            del self.__running_processes[index]
            running_application.cleanup()

    def launch_application(  # pylint: disable=too-many-arguments
            self,
            package_name: str,
            command: str,
            extra_arguments: typing.Iterable[typing.Any],
            leave_path_alone: bool,
            prefix: str,
            sender: str,  # pylint: disable=unused-argument
            non_conda: bool,
            app_type: constants.AppType,
    ) -> None:
        """
        Launch application from home screen.

        :param package_name: Name of the conda package, or alias of the external application.
        :param command: Exact command to launch application with.
        :param extra_arguments: Additional arguments to attach to command.
        :param prefix: Conda prefix, which should be active.
        :param app_type: Type of the application.
        """
        self.main_window.update_status(action=f'Launching <b>{package_name}</b>', value=0, max_value=0)

        def next_step(*args: typing.Any) -> None:  # pylint: disable=unused-argument
            self.__launch_application(
                package_name=package_name,
                command=command,
                extra_arguments=extra_arguments,
                leave_path_alone=leave_path_alone,
                prefix=prefix,
                non_conda=non_conda,
            )

        if app_type == constants.AppType.CONDA:
            next_step()

        elif app_type == constants.AppType.INSTALLABLE:
            app = external_apps.get_applications(cached=True).installable_apps[package_name]

            app.update_config(self.main_window.current_prefix)

            # Install extensions first!
            worker = app.install_extensions()  # pylint: disable=assignment-from-no-return
            worker.sig_finished.connect(next_step)
            worker.start()

        elif app_type == constants.AppType.WEB:
            webbrowser.open_new_tab(command)

            self.__feedback_timer.start()

    def __feedback_timeout(self) -> None:
        """
        Hide application checking progress bar.

        Actual feedback is done in :meth:`~ApplicationsComponent.update_running_applications`.
        """
        self.main_window.update_status()

    def __launch_application(  # pylint: disable=too-many-arguments
        self,
        package_name: str,
        command: str,
        extra_arguments: typing.Iterable[typing.Any],
        leave_path_alone: bool,
        prefix: str,
        non_conda: bool,
    ) -> None:
        """Second phase of the :meth:`~ApplicationComponent.launch_application`."""
        environment: typing.Dict[str, str] = dict(os.environ)
        environment.pop('QT_API')

        if anaconda_config.MAC:
            # See https://github.com/ContinuumIO/anaconda-issues/issues/3287
            os.environ['LANG'] = os.environ.get('LANG') or os.environ.get('LC_ALL') or 'en_US.UTF-8'
            os.environ['LC_ALL'] = os.environ.get('LC_ALL') or os.environ['LANG']

            # See https://github.com/ContinuumIO/navigator/issues/1233
            environment['EVENT_NOKQUEUE'] = '1'

        running_process: typing.Optional[launch.RunningProcess] = launch.launch(
            root_prefix=self.main_window.api.ROOT_PREFIX,
            prefix=prefix,
            command=command,
            extra_arguments=extra_arguments,
            package_name=package_name,
            environment=environment,
            leave_path_alone=leave_path_alone,
            non_conda=non_conda,
        )
        if running_process is None:
            return

        self.__running_processes.append(running_process)

        self.__feedback_timer.start()

    def show_application_launch_errors(self, application: launch.RunningProcess) -> None:
        """Show a dialog with details on application launch error."""
        self.main_window.update_status()
        if not self.main_window.config.get('main', 'show_application_launch_errors'):
            return

        content: typing.Optional[str] = application.stderr
        if content:
            content = content.strip()
        else:
            content = f'Exit code: {application.return_code}'

        telemetry.ANALYTICS.instance.event('navigate', {'location': '/home/launch_error'})
        dialogs.MessageBoxError(
            text=f'Application <b>{application.package}</b> launch may have produced errors.',
            title='Application launch error',
            error=content,
            report=False,
            learn_more=None,
        ).exec_()
        telemetry.ANALYTICS.instance.event('navigate', {'location': '/home'})

    def check_dependencies_before_install(  # pylint: disable=too-many-statements
            self, worker, output, error,  # pylint: disable=unused-argument
    ):
        """
        Check if the package to be installed changes navigator dependencies.

        This check is made for Orange3 which is not qt5 compatible.
        """
        if isinstance(output, dict):
            exception_type = str(output.get('exception_type', ''))
            actions = output.get('actions', {})
        else:
            exception_type = ''
            actions = {}

        conflicts = False
        nav_deps_conflict = self.main_window.api.check_navigator_dependencies(actions, self.main_window.current_prefix)
        conflict_message = ''

        # Try to install in a new environment
        if 'UnsatisfiableError' in exception_type or nav_deps_conflict:
            conflicts = True
            # Try to set the default python to None to avoid issues that
            # prevent a package to be installed in a new environment due to
            # python pinning, fusion for 2.7, rstudio on win for 2.7 etc.
            self.main_window.api.conda_config_set('default_python', None)

        if conflicts:
            telemetry.ANALYTICS.instance.event('navigate', {'location': '/conflict_creating_environment'})
            dlg = environment_dialogs.ConflictDialog(
                parent=self.main_window,
                package=worker.pkgs[0],
                extra_message=conflict_message,
                current_prefix=self.main_window.current_prefix,
            )
            self.main_window._dialog_environment_action = dlg
            worker_info = self.main_window.api.conda_data(prefix=self.main_window.current_prefix)
            worker_info.sig_chain_finished.connect(dlg.setup)

            if dlg.exec_():
                env_prefix = dlg.prefix
                action_msg = f'Installing application <b>{worker.pkgs[0]}</b> on newenvironment <b>{env_prefix}</b>'

                if env_prefix not in dlg.environments:
                    new_worker = self.main_window.api.create_environment(
                        prefix=env_prefix,
                        packages=worker.pkgs,
                        no_default_python=True,
                    )
                    # Save the old prefix in case of errors
                    new_worker.old_prefix = worker.prefix

                    new_worker.action_msg = action_msg
                    new_worker.sig_finished.connect(self.main_window._conda_output_ready)
                    new_worker.sig_partial.connect(self.main_window._conda_partial_output_ready)
                else:
                    new_worker = self.main_window.api.install_packages(
                        prefix=env_prefix,
                        pkgs=worker.pkgs,
                        no_default_python=True,
                    )
                    # Save the old prefix in case of errors
                    new_worker.old_prefix = worker.prefix

                    new_worker.action = constants.ACTION_INSTALL
                    new_worker.action_msg = action_msg
                    new_worker.pkgs = worker.pkgs
                    new_worker.sig_finished.connect(self.main_window._conda_output_ready)
                    new_worker.sig_partial.connect(self.main_window._conda_partial_output_ready)
                self.main_window.update_status(action_msg, value=0, max_value=0)
            else:
                self.main_window.set_widgets_enabled(True)
                self.main_window.set_busy_status(conda=False)
                self.main_window.update_status()

            self.main_window._dialog_environment_action = None
            telemetry.ANALYTICS.instance.event('navigate', {'location': '/environments'})
        else:
            if worker.action == constants.APPLICATION_INSTALL:
                action_msg = f'Install application <b>{worker.pkgs[0]}</b> on <b>{worker.prefix}</b>'
            elif worker.action == constants.APPLICATION_UPDATE:
                action_msg = f'Updating application <b>{worker.pkgs[0]}</b> on <b>{worker.prefix}</b>'
            new_worker = self.main_window.api.install_packages(
                prefix=worker.prefix,
                pkgs=worker.pkgs,
            )
            new_worker.action_msg = action_msg
            new_worker.action = worker.action
            new_worker.sender = worker.sender
            new_worker.non_conda = worker.non_conda
            new_worker.pkgs = worker.pkgs
            new_worker.sig_finished.connect(self.main_window._conda_output_ready)
            new_worker.sig_partial.connect(self.main_window._conda_partial_output_ready)
            self.main_window.update_status(action_msg, value=0, max_value=0)

    def check_license_requirements(self, worker, output, error):
        """Check if package requires licensing and try to get a trial."""
        worker.output = output
        self.check_dependencies_before_install(worker, output, error)

    def conda_application_action(  # pylint: disable=missing-function-docstring,too-many-arguments,too-many-statements
            self, action, package_name, version, sender, non_conda, app_type,
    ):
        if app_type == constants.AppType.INSTALLABLE and action == constants.APPLICATION_INSTALL:
            self.install_external_app(package_name)
            return

        self.main_window.tab_home.set_widgets_enabled(False)
        if 'environments' in self.main_window.components:
            self.main_window.components.environments.tab.set_widgets_enabled(False)
        self.main_window.set_busy_status(conda=True)
        current_version = self.main_window.api.conda_package_version(
            pkg=package_name,
            prefix=self.main_window.current_prefix,
        )

        if version:
            pkgs = [f'{package_name}=={version}']
        else:
            pkgs = [f'{package_name}']

        if action == constants.APPLICATION_INSTALL:
            worker = self.main_window.api.install_packages(
                prefix=self.main_window.current_prefix,
                pkgs=pkgs,
                dry_run=True,
            )
            text_action: str = 'Installing'

            if current_version:
                text_action = {
                    -1: 'Downgrading',
                    1: 'Upgrading',
                }.get(
                    version_utils.compare(version, current_version),
                    text_action,
                )

            action_msg = (
                f'{html.escape(text_action)} application <b>{html.escape(package_name)}</b> on '
                f'<b>{html.escape(self.main_window.current_prefix)}</b>'
            )
            worker.prefix = self.main_window.current_prefix
            worker.action = action
            worker.action_msg = action_msg
            worker.sender = sender
            worker.pkgs = pkgs
            worker.non_conda = non_conda
            worker.sig_finished.connect(self.check_license_requirements)
            worker.sig_partial.connect(self.main_window._conda_partial_output_ready)
        elif action == constants.APPLICATION_UPDATE:
            worker = self.main_window.api.install_packages(
                prefix=self.main_window.current_prefix,
                pkgs=pkgs,
                dry_run=True,
            )
            action_msg = (
                f'Updating application <b>{html.escape(package_name)}</b> on '
                f'<b>{html.escape(self.main_window.current_prefix)}</b>'
            )
            worker.prefix = self.main_window.current_prefix
            worker.action = action
            worker.action_msg = action_msg
            worker.sender = sender
            worker.pkgs = pkgs
            worker.non_conda = non_conda
            worker.sig_finished.connect(self.check_license_requirements)
            worker.sig_partial.connect(self.main_window._conda_partial_output_ready)
        elif action == constants.APPLICATION_REMOVE:
            worker = self.main_window.api.remove_packages(prefix=self.main_window.current_prefix, pkgs=pkgs)
            action_msg = (
                f'Removing application <b>{html.escape(package_name)}</b> from '
                f'<b>{html.escape(self.main_window.current_prefix)}</b>'
            )
            worker.action = action
            worker.action_msg = action_msg
            worker.sender = sender
            worker.pkgs = pkgs
            worker.non_conda = non_conda
            worker.sig_finished.connect(self.main_window._conda_output_ready)
            worker.sig_partial.connect(self.main_window._conda_partial_output_ready)
        self.main_window.update_status(action_msg, value=0, max_value=0)

    @staticmethod
    def install_external_app(app_name):
        """Installing external app (VSCode, pycharm)."""
        external_apps.get_applications(cached=True).installable_apps[app_name].install()

    def setup(self, worker: typing.Any, output: typing.Any, error: str, initial: bool) -> None:
        """Perform component configuration from `conda_data`."""
        prefix: str
        for prefix in output['processed_info']['__environments']:
            launch.remove_package_logs(root_prefix=self.main_window.api.ROOT_PREFIX, prefix=prefix)
